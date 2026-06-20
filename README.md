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

鈹�

鈹溾攢鈹€ client1\_bert.py

鈹溾攢鈹€ client2\_bert.py

鈹溾攢鈹€ server.py

鈹溾攢鈹€ model/

鈹�   鈹溾攢鈹€ train\_bert.py

鈹�   鈹溾攢鈹€ evaluate.py

鈹�   鈹斺攢鈹€ predict.py

鈹�

鈹溾攢鈹€ data/

鈹�   鈹溾攢鈹€ scam\_messages.csv

鈹�   鈹溾攢鈹€ client1/

鈹�   鈹斺攢鈹€ client2/

鈹�

鈹溾攢鈹€ results/

鈹溾攢鈹€ docs/

鈹斺攢鈹€ README.md

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



