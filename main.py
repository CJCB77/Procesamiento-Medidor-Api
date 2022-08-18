from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from skimage.io import imread
import cv2
import boto3
from os import getenv,remove
import numpy as np
from dotenv import load_dotenv


app = FastAPI()

load_dotenv()


class PostImage(BaseModel):
    image: str


@app.get("/")
def read_root():
    return {"Message": "Bienvenido a la app de procesamiento de contornos"}

@app.post("/filter")
async def read_image(post_image: PostImage):
    image_path = post_image.image
    image_medidor = imread(image_path)

    #Tranform into grayscale
    image_medidor_gray = cv2.cvtColor(image_medidor, cv2.COLOR_BGR2GRAY)
    # APPLY GAUSSIAN BLUR
    image_medidor_blur = cv2.GaussianBlur(image_medidor_gray, (95, 95), 0)
    # Apply morphological gradient
    # image_medidor_gradient = cv2.morphologyEx(image_medidor_blur, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    # divide gray by morphology image
    division = cv2.divide(image_medidor_gray, image_medidor_blur, scale=192)

   
    #Apply threshold
    image_medidor_threshold = cv2.threshold(division, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


    #Write image_medidor_threshold in images folder
    #Format datetime
    now = datetime.now()
    current_time = now.strftime("%d-%m-%Y_%H-%M-%S")
    img_name = f'{current_time}image_medidor_threshold.png'
    cv2.imwrite(f"images/{img_name}", image_medidor_threshold)
    img_path = f'images/{img_name}'

    #Add img to S3 bucket
    s3 = boto3.resource('s3', 
        aws_access_key_id=getenv('AWS_ACCESS_KEY_ID'), 
        aws_secret_access_key=getenv('AWS_SECRET_ACCESS_KEY'))

    #Use AWS_BUCKET from .env
    bucket = s3.Bucket(getenv('AWS_BUCKET'))
    bucket.upload_file(img_path, img_name)
    
    uploaded_file_url = f"https://{getenv('AWS_BUCKET')}.s3.amazonaws.com/{img_name}"
    print(uploaded_file_url)

    #Remove img from images folder
    remove(img_path)

    return {"message": "Imagen recibida", "url": uploaded_file_url}
