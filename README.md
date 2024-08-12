# Retro_Project1

This project can detect a variety of hand signs and display what each hand sign is to the user. The user can make a sign with their hand on the camera, and the project will use machine learning to detect which of the nine possible hand signs it is.

![Screenshot 2024-08-12 114206](https://github.com/user-attachments/assets/1c1d5b97-354d-407c-a7da-c48fb424ebf3)


## The Algorithm

Inside of the while loop, it parses the camera feed into individual picture frames, and each frame gets classified and then the loop moves on to the next frame. Each class is made up with around 200 image samples that I took myself using my own hand.

## Running this project

1. Install the Jetson util and Jetson Inference libraries.
2. Clone the project from my github page and put it into VS Code. 
3. Connect to a camera with your nano.
4. Type "cd project" into the terminal to navigate to the directory of the project.
5. Run this command to start the project: python3  rishik.py --network=models/resnet18.onnx --labels=models/labels.txt --input_blob=input_0 --output_blob=output_0 --ssl-key=key.pem --ssl-cert=cert.pem /dev/video0 webrtc://@:8554/my_output
6. Type https://(your ip address):8554 to go to the camera, and put your hand signs in front of the camera to use the project.

View a video explanation here: https://drive.google.com/file/d/16ny4kKBbPTUbmHsvR0iSeygA5QGdowkv/view?usp=sharing
