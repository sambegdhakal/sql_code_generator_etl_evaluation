FROM python:3.11-slim

# setting up a work directory as sql_generation_project
WORKDIR /sql_generation_project

#copy everything to the code
COPY . /sql_generation_project

#install dependencies from requirements
RUN  pip install --no-cache-dir -r requirements.txt

# Defult command
CMD ["python", "main.py"]
