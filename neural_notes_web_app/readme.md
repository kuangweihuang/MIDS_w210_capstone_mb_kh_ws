# Neural Notes
### Visual Exploration of Songs using AI Content-Based Embeddings

## 1. Running Neural Notes on Google App Engine
- Visit the link: [https://neural-notes.appspot.com/](https://neural-notes.appspot.com/)

## 2. Running Neural Notes on your localhost
- Clone this GitHub repo

- Navigate ot this directory on your local machine

- Activate the virtual environment using the following bash command:

	`$ source venv/Scripts/activate `

- Launch a python server with the Flask app `main.py`:

	`$ python main.py`

- Open a web browser (either Firefox or Chrome) and paste in the `http://` address in the bash prompt:

	`$ http://127.0.0.1:8050/`


#### Note: 
There are two versions, the original `main.py` version and a separate version with some famous songs fit into the embedding space `main_w_famous.py`. Simply rename the desired version to be `main.py` and run the code above.
