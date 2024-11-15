import torch



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res



def evaluate(model, data_loader, criterion):    
    device = next(model.parameters()).device
    model.eval()
    num_videos = len(data_loader.dataset.samples)
    # num_classes = len(data_loader.dataset.classes)
    agg_preds = torch.zeros((num_videos, 2), dtype=torch.float32, device=device)
    agg_targets = torch.zeros((num_videos), dtype=torch.int32, device=device)
    agg_preds = []
    agg_targets = []
    idx = 0
    # with torch.no_grad():
    with torch.inference_mode():
        for video, target in data_loader:
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            # Use softmax to convert output into prediction probability
            preds = torch.softmax(output, dim=1)
            for b in range(video.size(0)):
                # idx = video_idx[b].item()
                # agg_preds[idx] += preds[b].detach()
                # agg_targets[idx] = target[b].detach().item()
                agg_preds.append(preds[b].detach())
                agg_targets.append(target[b].detach())
                
        agg_preds = torch.stack(agg_preds)
        agg_targets = torch.stack(agg_targets)
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # acc1, acc2 = accuracy(output, target, topk=(1, 2))
        acc1, acc2 = accuracy(agg_preds, agg_targets, topk=(1, 2))
        
        results = {
            'accuracy': acc1.item(),
            'acc2': acc2.item(),
            'loss': loss.item()
        }

    return results

if __name__ == '__main__':
    from dataset_loader import Dataset_test
    test_dataset = Dataset_test(num_segments=32)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False)
    from model import Model
    model = Model(2048)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    results = test(model, test_loader)

    gt = results['groundtruth']
    pred = results['prediction']
    gt_class = gt.astype(int)
    import numpy as np
    probs = np.arange(0, 1, 0.001)

    from sklearn.metrics import accuracy_score

    best_acc = 0
    best_threshold = 0
    for prob in probs:        
        boolean_value = pred >= prob
        pred_class = boolean_value.astype(int)
        acc = accuracy_score(gt_class, pred_class)
        if acc > best_acc:
            best_acc = acc
            best_threshold = prob
    
    print('end')

'''
def evaluate(model, test_loader, criterion):
    device = next(model.parameters()).device
    model.eval()


    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, label)
        
            probs = torch.nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            



def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    # Group and aggregate output of a video
    num_videos = len(data_loader.dataset.samples)
    num_classes = len(data_loader.dataset.classes)
    agg_preds = torch.zeros((num_videos, num_classes), dtype=torch.float32, device=device)
    agg_targets = torch.zeros((num_videos), dtype=torch.int32, device=device)
    with torch.inference_mode():
        for video, target, video_idx in metric_logger.log_every(data_loader, 100, header):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            # Use softmax to convert output into prediction probability
            preds = torch.softmax(output, dim=1)
            for b in range(video.size(0)):
                idx = video_idx[b].item()
                agg_preds[idx] += preds[b].detach()
                agg_targets[idx] = target[b].detach().item()

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if isinstance(data_loader.sampler, DistributedSampler):
        # Get the len of UniformClipSampler inside DistributedSampler
        num_data_from_sampler = len(data_loader.sampler.dataset)
    else:
        num_data_from_sampler = len(data_loader.sampler)

    if (
        hasattr(data_loader.dataset, "__len__")
        and num_data_from_sampler != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the sampler has {num_data_from_sampler} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(
        " * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5
        )
    )
    # Reduce the agg_preds and agg_targets from all gpu and show result
    agg_preds = utils.reduce_across_processes(agg_preds)
    agg_targets = utils.reduce_across_processes(agg_targets, op=torch.distributed.ReduceOp.MAX)
    agg_acc1, agg_acc5 = utils.accuracy(agg_preds, agg_targets, topk=(1, 5))
    print(" * Video Acc@1 {acc1:.3f} Video Acc@5 {acc5:.3f}".format(acc1=agg_acc1, acc5=agg_acc5))
    return metric_logger.acc1.global_avg

'''