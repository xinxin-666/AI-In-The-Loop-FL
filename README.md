\# AI-In-The-Loop-FL



Federated Learning based scam message detection system using BERT with Differential Privacy.



\## Project Overview



This project implements a federated learning framework for scam message detection using BERT and Flower. Multiple clients collaboratively train a global model without sharing raw data.



Features:



\* BERT text classification

\* Federated Learning (Flower)

\* Differential Privacy (Opacus)

\* Scam message detection

\* Distributed client-server architecture



\---



\## Project Structure



```text

AI-In-The-Loop-FL

│

├── client1\_bert.py

├── client2\_bert.py

├── server.py

├── model/

│   ├── train\_bert.py

│   ├── evaluate.py

│   └── predict.py

│

├── data/

│   ├── scam\_messages.csv

│   ├── client1/

│   └── client2/

│

├── results/

├── docs/

└── README.md

```



\## Environment Setup



Install dependencies:



```bash

pip install -r requirements.txt

```



\## Run Steps



Start server:



```bash

python server.py

```



Open another terminal:



```bash

python client1\_bert.py

```



Open another terminal:



```bash

python client2\_bert.py

```



\## Training Results



Federated learning training completed successfully.



Rounds: 10



Loss:



\* Round 1: 0.10

\* Round 5: 0.10

\* Round 10: 0.10



\## Team Contributions



Student A:



\* Data preprocessing

\* Dataset cleaning



Student B:



\* Model design



Student C:



\* Federated learning implementation



Student Xinxin:



\* BERT training modification

\* Federated learning debugging

\* Git integration and deployment



