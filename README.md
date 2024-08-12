# HuggingFace: https://huggingface.co/rakib72642/Face_Detection_Logic

# Setup Global API

sudo apt install iproute2 -y && sudo apt install wget -y && sudo apt install unzip -y && sudo apt install unzip -y && sudo apt install nvtop -y && sudo apt-get install git-all -y && sudo apt-get install git-lfs -y && sudo apt-get update && sudo apt-get install libgl1 -y && sudo apt install curl -y && curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok -y && sudo apt update && sudo apt upgrade -y && ngrok config add-authtoken 2Qm8hS1zPhVXiLjEdlI4738tLzF_2QJwGJMK5oTbQD33QSVXS && ngrok http --domain=he.ngrok.app 8060

# Setup Local API

git clone https://huggingface.co/rakib72642/Face_Detection_Logic && cd Face_Detection_Logic && pip install -r requirements.txt && sudo apt update && sudo apt upgrade -y && python face_logic_api.py

cd Face_Detection_Logic && python face_logic_api.py

# hypercorn face_logic_api:app --bind 127.0.0.1:8060 --workers 4
