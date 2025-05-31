# 🧠 Disease Prediction Dashboard (Big Data Project)

This project is a disease prediction system using Apache Spark, HDFS, and Streamlit. It processes medical data for training and testing using a Spark ML pipeline and visualizes predictions and insights through an interactive web dashboard.

## 🚀 Quick Start

### 1. Start the Docker Containers

Make sure Docker and Docker Compose are installed, then run:

```bash
docker-compose up -d
````

### 2. Load Training and Testing CSV Files to HDFS

CSV files are located in the `Makefile/` directory.

In your terminal (inside the namenode container), run:

```bash
hdfs dfs -mkdir -p /data
hdfs dfs -put Makefile/Training.csv /data/
hdfs dfs -put Makefile/Testing.csv /data/
```

> 💡 Adjust the paths above if you're mounting volumes differently.

### 3. Run the Spark Job

In a new terminal, run your Spark job:

```bash
docker exec -it bigdataproject-spark-1 spark-submit /opt/spark/diseasePrediction.py
```

### 4. Run the Visuals Script (Optional)

```bash
docker exec -it bigdataproject-spark-1 spark-submit /opt/spark/visuals.py
```

### 5. Run the Streamlit Dashboard

Launch the web interface:

```bash
docker exec -it bigdataproject-spark-1 streamlit run /opt/spark-app/app.py --server.port=4041 --server.address=0.0.0.0
```

### 6. Open in Browser

Visit [http://localhost:4041](http://localhost:4041) to view your Streamlit dashboard.

---

## 📁 Project Structure

```
Makefile/
├── Training.csv
├── Testing.csv
spark-app/
├── diseasePrediction.py
├── visuals.py
├── app.py
docker-compose.yaml
config
README.md
```

---

## 🛠️ Tech Stack

* Apache Spark
* HDFS
* Streamlit
* Docker

---

## ✍️ Author

Made with ❤️ by \[Parjeet Mongar and Keshar Bhujel]
