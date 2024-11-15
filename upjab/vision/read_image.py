import os.path as osp
from PIL import Image

def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img


if __name__ == '__main__':
    path = 'example_data/images/disease_image/images_original/119799_objt_rs_2020-12-15_13-14-02-33_002.JPG'
    image = read_image(path)
    print('End')