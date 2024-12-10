s = 'There are multiple fish in the image'


from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
config = "/home/ray/workspace/codes/projects/upjab/upjab/other/GroundingDINO_SwinT_OGC.py"
pth_path = "/home/ray/workspace/codes/projects/upjab/upjab/other/groundingdino_swint_ogc.pth"
model = load_model(config, pth_path)

def detect(
    IMAGE_PATH,
    TEXT_PROMPT="cat . dog .",
    BOX_TRESHOLD=0.35,
    TEXT_TRESHOLD=0.25,
    device="cpu"
    ):

    image_source, image = load_image(IMAGE_PATH)

    file_name = os.path.basename(IMAGE_PATH)  # name of file
    folder_path = os.path.dirname(IMAGE_PATH)  # folder of file

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=device
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(f"{folder_path}/annotated_{file_name}", annotated_frame)


if __name__ == "__main__":
    image_path = 'example_data/images/assets/astronaut.jpg'
    detect(image_path,TEXT_PROMPT="astronaut .")
    print('done')