The file build and deploy contains all the files needed to deploy the dashboard

Follow the commands on windows Powershell Prompt to build and deploy this dashboard :

0. Set the repo to ./build and deploy
1. docker compose up 
2. To open the dashboard use the link : http://localhost:8501 or http://0.0.0.0:8501

You can run the command :

docker system prune -a -f : To delete all containers and images
docker iamges : To see all images created
docker build -t nameofimage . : To build an image from a DockerFile
docker run -p 8501:8501 nameofimage : To run an image

