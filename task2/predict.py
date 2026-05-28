import numpy as np
import cv2
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=-1, det_size=(640, 640))

THRESHOLD = 0.55


def decode_image(image_bytes):
    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError("Could not decode image — file may be corrupt or not a valid image.")
    return image


def detect_face(image):
    faces = model.get(image)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def extract_embedding(face):
    return face.normed_embedding



def get_bounding_box(face):
    x1, y1, x2, y2 = face.bbox.astype(int)
    return [int(x1), int(y1), int(x2), int(y2)]


def calculate_similarity(embedding1, embedding2):
    return float(cosine_similarity([embedding1], [embedding2])[0][0])


def get_verification_result(similarity):
    if similarity >= THRESHOLD:
        return "same person"
    return "different person"


def predict(image1_bytes, image2_bytes):
    try:
        image1 = decode_image(image1_bytes)
        image2 = decode_image(image2_bytes)
    except ValueError as e:
        return {"error": str(e)}

    face1 = detect_face(image1)
    face2 = detect_face(image2)

    if face1 is None:
        return {"error": "No face detected in image 1"}
    if face2 is None:
        return {"error": "No face detected in image 2"}

    embedding1 = extract_embedding(face1)
    embedding2 = extract_embedding(face2)

    similarity = calculate_similarity(embedding1, embedding2)

    verification_result = get_verification_result(similarity)

    bounding_box_image1 = get_bounding_box(face1)
    bounding_box_image2 = get_bounding_box(face2)

    return {
        "verification_result": verification_result,
        "similarity_score": round(similarity, 4),
        "bounding_box_image1": bounding_box_image1,
        "bounding_box_image2": bounding_box_image2,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python predict.py <image1_path> <image2_path>")
        sys.exit(1)

    img1_bytes = open(sys.argv[1], "rb").read()
    img2_bytes = open(sys.argv[2], "rb").read()

    result = predict(img1_bytes, img2_bytes)
    for k, v in result.items():
        print(f"  {k}: {v}")