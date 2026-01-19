ETL SQL Generator Docker Image

⚠️ Important: This container requires Ollama (or your LLaMA/Groq host) running locally on port 11434. Make sure your Ollama server is running and accessible before using the container.

This Docker image contains a Python-based ETL SQL generator. It comes preloaded with a default input file (Transformation_logic.xlsx) so you can immediately run transformations and generate output locally.

The script supports environment variables for input and output files, making it flexible when mounting folders in Docker.

✅ Features

Preloaded Transformation_logic.xlsx for quick testing.

Generates SQL transformations from the input Excel file.

Fully containerized; no need to install Python or dependencies locally.

Supports custom input/output files via environment variables.

Versioned Docker image (latest and 1.0.0) for reproducibility.

📦 Pull the Docker Image
docker pull <YOUR_DOCKERHUB_USERNAME>/etl_sql_generator:latest


Replace <YOUR_DOCKERHUB_USERNAME> with your Docker Hub username.

🚀 Run the Container (Windows)

Use this single-line command to generate SQL outputs:

docker run --rm -e LLM_HOST=host.docker.internal -e LLM_PORT=11434 -e OUTPUT_FILE=/app/output/transformed_with_sql.xlsx -v C:\Users\sambe\Desktop\etl_output:/app/output etl_sql_generator:latest


Explanation:

--rm → automatically removes the container after execution.

-e LLM_HOST / -e LLM_PORT → connect to your local LLaMA/Groq host.

-e OUTPUT_FILE → path inside the container where the output will be written.

-v C:\Users\sambe\Desktop\etl_output:/app/output → mount your local folder so outputs appear there.

Default input inside the container:
/app/transformation_files/Transformation_logic.xlsx

Default output inside the container (if no custom OUTPUT_FILE):
/app/transformation_files/transformed_with_sql.xlsx

📝 Using Your Own Input File

If you want to use a custom Excel file:

Place your Excel file in a local folder, e.g., C:\Users\sambe\Desktop\etl_input.

Mount it inside the container and set environment variables:

docker run --rm -e LLM_HOST=host.docker.internal -e LLM_PORT=11434 -v C:\Users\sambe\Desktop\etl_input:/app -v C:\Users\sambe\Desktop\etl_output:/app/output -e INPUT_FILE="/app/Transformation_logic.xlsx" -e OUTPUT_FILE="/app/output/transformed_with_sql.xlsx" etl_sql_generator:latest


Notes:

INPUT_FILE → path to your input Excel inside the container.

OUTPUT_FILE → path to write generated SQL inside the container.

⚡ Modify Input File

Open Transformation_logic.xlsx in Excel or Google Sheets.

Edit or add transformation rules as needed.

Save the file and rerun the container using the steps above.

📂 Output

All generated SQL scripts or outputs will be saved in the folder you mounted as /app/output.
You can open these files directly on your local machine.

📌 Notes

The container uses Python 3.11 with all dependencies pre-installed.

Docker must be installed and running locally.

By default, the container uses the preloaded Transformation_logic.xlsx if no custom file is provided.

Versioned Image:

docker pull <YOUR_DOCKERHUB_USERNAME>/etl_sql_generator:1.0.0
docker run --rm -v C:\Users\sambe\Desktop\etl_output:/app/output <YOUR_DOCKERHUB_USERNAME>/etl_sql_generator:1.0.0