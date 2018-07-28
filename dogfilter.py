import cv2

def apply_mask(face, mask):
    mask_h, mask_w, _ = mask.shape
    face_h, face_w, _ = face.shape

    factor = min(face_h / mask_h, face_w / mask_w)
    new_mask_w = int(factor * mask_w)
    new_mask_h = int(factor * mask_h)
    resized_mask = cv2.resize(mask, (new_mask_w, new_mask_h))

    face_with_mask = face.copy()
    non_white_pixels = (resized_mask < 250).all(axis=2)
    off_h = int((face_h - new_mask_h) / 2)
    off_w = int((face_w - new_mask_w) / 2)
    face_with_mask[off_h: off_h+new_mask_h, off_w:off_w+new_mask_w] [non_white_pixels] = resized_mask[non_white_pixels]

    return face_with_mask

def main():
    cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")
    frame = cv2.imread("assets/children.png")
    mask = cv2.imread("assets/sheepdog.png")
    rectangles = cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x,y,w,h) in rectangles:
        face = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = apply_mask(face, mask)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite("out.png", frame)
main()
