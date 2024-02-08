import face_recognition
import cv2
from typing import List
import os
import numpy as np
import insightface
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException,Request
from fastapi.responses import HTMLResponse,RedirectResponse
from starlette.status import HTTP_302_FOUND
import uvicorn 
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from PIL import Image, ImageDraw
from pathlib import Path
import logging
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


templates = Jinja2Templates(directory="view")

logger = logging.getLogger("uvicorn")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")


# insightface 얼굴 분석기 초기화
app_insightface = FaceAnalysis()
app_insightface.prepare(ctx_id=0, det_size=(640, 640))

# 새로운 이미지 분석
def is_human_face(file_path):
    img = cv2.imread(str(file_path))  # OpenCV를 사용하여 이미지 로드
    if img is None:
        logger.info(f"Image load failed: {file_path}")
        return False  # 이미지 로드 실패시 False 반환
    faces = app_insightface.get(img)
    logger.info(f"Detected {len(faces)} faces in {file_path}")
    return len(faces) > 0

#사람 얼굴 특징을 학습
ka_images_paths = ["./static/ka/ka.jpg", "./static/ka/ka2.jpg", "./static/ka/ka3.jpg"] # 'ka'의 이미지 경로 리스트
ka_encodings = []

for img_path in ka_images_paths:
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        ka_encodings.extend(encodings)
        
a_images_paths = ["./static/yoon/yoon.jpg", "./static/yoon/yoon2.jpg", "./static/yoon/yoon3.jpg", "./static/yoon/yoon4.jpg", "./static/yoon/yoon5.jpg", "./static/yoon/yoon6.jpg"] # 'yoon'의 이미지 경로 리스트
a_encodings = []

for img_path in a_images_paths:
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        a_encodings.extend(encodings)

# ka_image = face_recognition.load_image_file("./static/ka.jpg")
# ka_face_encoding = face_recognition.face_encodings(ka_image)[0]

# # 다른 사람의 얼굴 특징도 학습
# person_b_image = face_recognition.load_image_file("./static/yoon.jpg")
# person_b_face_encoding = face_recognition.face_encodings(person_b_image)[0]


c_images_paths = ["./static/ni/ni.jpg"] # 'ni'의 이미지 경로 리스트
c_encodings = []

for img_path in c_images_paths:
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        c_encodings.extend(encodings)

# person_c_image = face_recognition.load_image_file("./static/ni.jpg")
# person_c_face_encoding = face_recognition.face_encodings(person_c_image)[0]


d_images_paths = ["./static/minji/minji.jpg","./static/minji/minji1.jpg","./static/minji/minji2.png","./static/minji/minji3.jpg"] # 'minji'의 이미지 경로 리스트
d_encodings = []

for img_path in d_images_paths:
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        d_encodings.extend(encodings)

# person_d_image = face_recognition.load_image_file("./static/minji.jpg")
# person_d_face_encoding = face_recognition.face_encodings(person_d_image)[0]

e_images_paths = ["./static/one/one.jpg","./static/one/one2.jpg","./static/one/one3.jpg","./static/one/one4.jpg"] # 'one'의 이미지 경로 리스트
e_encodings = []

for img_path in e_images_paths:
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        e_encodings.extend(encodings)

# person_e_image = face_recognition.load_image_file("./static/one.jpg")
# person_e_face_encoding = face_recognition.face_encodings(person_e_image)[0]

f_images_paths = ["./static/he/he.jpg","./static/he/he2.jpg","./static/he/he3.jpg"] # 'one'의 이미지 경로 리스트
f_encodings = []

for img_path in f_images_paths:
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        f_encodings.extend(encodings)

# person_f_image = face_recognition.load_image_file("./static/he.jpg")
# person_f_face_encoding = face_recognition.face_encodings(person_f_image)[0]

g_images_paths = ["./static/jang/jang.jpg","./static/jang/jang2.jpg","./static/jang/jang3.jpg"] # 'one'의 이미지 경로 리스트
g_encodings = []

for img_path in g_images_paths:
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        g_encodings.extend(encodings)

# person_g_image = face_recognition.load_image_file("./static/jang.jpg")
# person_g_face_encoding = face_recognition.face_encodings(person_g_image)[0]

h_images_paths = ["./static/go/go.jpg","./static/go/go2.jpg","./static/go/go3.jpeg"] # 'one'의 이미지 경로 리스트
h_encodings = []

for img_path in g_images_paths:
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        h_encodings.extend(encodings)
        
i_images_paths = ["./static/ma/ma.jpg"] # 'one'의 이미지 경로 리스트
i_encodings = []

for img_path in i_images_paths:
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        i_encodings.extend(encodings)
        
j_images_paths = ["./static/what/what.jpg"] # 'one'의 이미지 경로 리스트
j_encodings = []

for img_path in j_images_paths:
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        j_encodings.extend(encodings)
        
k_images_paths = ["./static/emma/emma.jpg"] # 'one'의 이미지 경로 리스트
k_encodings = []

for img_path in k_images_paths:
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        k_encodings.extend(encodings)

# 인물 이름과 얼굴 특징을 매핑하는 딕셔너리 생성
face_encodings_to_names = {
    **{tuple(encoding): "ka" for encoding in ka_encodings},
    **{tuple(encoding): "yoon" for encoding in a_encodings},
    **{tuple(encoding): "ni" for encoding in c_encodings},
    **{tuple(encoding): "minji" for encoding in d_encodings},
    **{tuple(encoding): "one" for encoding in e_encodings},
    **{tuple(encoding): "he" for encoding in f_encodings},
    **{tuple(encoding): "jang" for encoding in g_encodings},
    **{tuple(encoding): "go" for encoding in h_encodings},
    **{tuple(encoding): "ma" for encoding in i_encodings},
    **{tuple(encoding): "what" for encoding in j_encodings},
    **{tuple(encoding): "emma" for encoding in k_encodings},
}

known_faces = list(face_encodings_to_names.keys())

@app.get("/",response_class=HTMLResponse)
async def main(request : Request):

    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def create_upload_file(files: List[UploadFile] = File(...)):
    UPLOAD_DIRECTORY = "./uploads"
    results = []
    
    logger.info(f"Received {len(files)} files")

    
    for file in files:
        file_path = Path(UPLOAD_DIRECTORY) / file.filename
        
        logger.info(f"Processing file: {file.filename}")

        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 얼굴 인식 및 분류 로직을 각 파일에 대해 수행
        if is_human_face(file_path):
            new_image = face_recognition.load_image_file(file_path)
            new_face_encodings = face_recognition.face_encodings(new_image)
            if new_face_encodings:
                new_face_encoding = new_face_encodings[0]
                matches = face_recognition.compare_faces(known_faces, new_face_encoding)
                logger.info(f"Matches for {file.filename}: {matches}")
                        
                if True in matches:
                    first_match_index = matches.index(True)
                    matched_face_encoding = known_faces[first_match_index]
                    person_name = face_encodings_to_names[tuple(matched_face_encoding)]
                
                    logger.info(f"Recognized {person_name} for file {file.filename}")
                
                    target_directory = Path(f"./images/{person_name}")
                else :
                    person_name = "unknown"
                    target_directory = Path(f"./images/{person_name}")
            else : 
                person_name = "unknown"
                target_directory = Path(f"./images/{person_name}")
            if not target_directory.exists():
                logger.info(f"Creating directory: {target_directory}")
                target_directory.mkdir(parents=True, exist_ok=True)
            else:
                logger.info(f"Directory already exists: {target_directory}")
            target_file_path = target_directory / file.filename
            try:
                shutil.move(str(file_path), target_file_path)
                logger.info(f"File {file.filename} moved to {target_directory}")
                results.append({"filename": file.filename, "status": "success", "message": f"얼굴이 인식되어 {person_name} 폴더에 저장되었습니다."})
            except Exception as e:
                logger.error(f"Failed to move file {file.filename} to {target_directory}: {e}")
                results.append({"filename": file.filename, "status": "failure", "message": "파일 이동 중 오류가 발생했습니다."})
        
    return results

selected_image_info = {}

@app.get("/select", response_class=HTMLResponse)
async def get_select(request: Request):
    # selected_image_info 글로벌 변수에서 이미지 정보 가져오기
    return templates.TemplateResponse("select.html", {
        "request": request,
        "selected_images": selected_image_info  # 이미지 정보 리스트를 직접 전달
    })

@app.get("/select", response_class=HTMLResponse)
async def select_page(request: Request):
    
    image_info = selected_image_info
    
    template_name = "select_result.html" if image_info else "select.html"


    
    return templates.TemplateResponse(template_name, {
        "request": request,
        "image_path": image_info.get('path', ''),
        "image_name": image_info.get('name', 'No image selected')
    })

@app.post("/select")
async def handle_select(request: Request, file: UploadFile = File(...)):
    TEMP_DIR = Path("./temp")
    TEMP_DIR.mkdir(exist_ok=True)
    temp_file_path = TEMP_DIR / file.filename

    # 업로드된 파일을 임시 디렉토리에 저장
    contents = await file.read()
    with open(temp_file_path, "wb") as f:
        f.write(contents)

    # 업로드된 이미지의 얼굴 특징 추출
    uploaded_image = face_recognition.load_image_file(temp_file_path)
    uploaded_face_encodings = face_recognition.face_encodings(uploaded_image)

    if not uploaded_face_encodings:
        return templates.TemplateResponse("index.html", {"request": request, "error": "얼굴을 인식할 수 없습니다."})

    uploaded_face_encoding = uploaded_face_encodings[0]
    most_similar_images = find_most_similar_images(uploaded_face_encoding[0])  # 첫 번째 얼굴 인코딩 사용

    global selected_image_info
    selected_image_info = [{
        'path': f"/images/{name}/{image_path.name}",
        'name': name
    } for image_path, name in most_similar_images]

    return RedirectResponse(url="/select", status_code=HTTP_302_FOUND)

    
def find_most_similar_images(uploaded_face_encoding, similarity_threshold=0.1):
    IMAGE_DIR = Path("./images")
    similar_images  = []

    # 모든 폴더와 이미지를 순회하며 비교
    for person_dir in IMAGE_DIR.iterdir():
        if person_dir.is_dir():
            for image_path in person_dir.iterdir():
                known_image = face_recognition.load_image_file(image_path)
                known_face_encodings = face_recognition.face_encodings(known_image)

                if known_face_encodings:
                    # 첫 번째 얼굴 인코딩만 사용하여 거리 계산
                    face_distance = face_recognition.face_distance(known_face_encodings, uploaded_face_encoding)[0]
                    # 유사도 임계값보다 낮은 경우에만 리스트에 추가
                    if face_distance < similarity_threshold:
                        similar_images.append((image_path, person_dir.name, face_distance))

    # 유사도(거리)에 따라 정렬 후 상위 결과 반환
    similar_images.sort(key=lambda x: x[2])
    return [(image[0], image[1]) for image in similar_images]

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000,reload=True)