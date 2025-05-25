1. Environment Setup:

    Using anaconda and local cuda gpu for traiing

        conda create -n chatbot python=3.10 -y
        conda activate chatbhot
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

    Setup python requirements using pip

        pip install -r requirements.txt

2. Prepare dataset for training:

    Requirement- dataset.json

        python datasetPrep.py 

    (this will generate train.csv and label_to_answer.json files)

3. Test CUDA: 

        python cudaTest.py

4. Train model (already trained and saved in models/ directory):

        python train.py

5. Run inference in CLI

        python chatbot.py

    NOTE: Model overall is not very good, partly due to limited dataset and unoptimised training args. GPT based model would have performed better but with the downside of requiring more resources and training time.

6. Flask App

    -Run app

        python app.py

    -test api endpoint

        curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d "{\"query\":\"What is PPE detection?\"}"

