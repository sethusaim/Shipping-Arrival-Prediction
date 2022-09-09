# Project: Shipping Cost Prediction

### Shipping Cost Prediction using Machine Learning Algorithms

In this data science project, you will build a machine learning system which will be able predict the cost of the shipment or package by using machine learning algorithms. This project will be very usefull for logistics companies, where on a day to day basis a lot of couriers,packages or goods are transported via different modes of transport. This main concern with these logistics companies, is trying to deliver these goods in an efficient and cost efficient way as possible, so pricing of the shipment is tricky and involves a lot of variables to consider while pricing of the shipment. There might be scenarios where the shipment might be delayed due to some external reasons, leading to loss for the company and delay in delivery of the shipment. So logistics companies need to use dynamic pricing based on several factors and variables to price the shipment in such a way that there are no loss to the company and price of the shipment is as less as possible so that customers can use thier services more due to effective pricing rates.

Now the question is how to dynamically predict prices of the particular shipment ?. One of the approaches which we can use of machine learning approach, where we can predict the shipping price based on the domain knowledge and leverage previous shipment data to predict the prices. 


## Tech Stack Used
1. Python 
2. FastAPI 
3. Machine learning algorithms
4. Docker
%. MongoDB

## How to run?
Before we run the project, make sure that you are having MongoDB in your local system, with Compass since we are using MongoDB for data storage.

### Step 1: Clone the repository
```bash
git clone https://github.com/sethusaim/Shipping-Arrival-Prediction.git
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -n ship python=3.7.6 -y
```

```bash
conda activate ship
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Export the MongoDB URL environment variable
```bash
export MONGODB_URL="mongodb://localhost:27017"
```

### Step 5 - Run the application server
```bash
python app.py
```