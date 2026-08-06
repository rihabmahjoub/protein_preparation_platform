FROM continuumio/miniconda3:latest
WORKDIR /app
RUN conda install -y -c conda-forge python=3.10 pdbfixer openmm \
    && conda clean --all -y
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
