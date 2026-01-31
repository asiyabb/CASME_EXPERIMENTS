{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f0ffb930-44cc-4d3e-babb-dd73236002d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, time\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import cv2\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torchvision import transforms\n",
    "from torchvision.models import resnet18, ResNet18_Weights\n",
    "\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "91a959da-2a9c-45f7-a6ac-bd2f787f1f44",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Device: cpu\n"
     ]
    }
   ],
   "source": [
    "# ===== EDIT THESE PATHS =====\n",
    "DATA_ROOT = r\"C:\\Users\\ASIF\\CASME DATASET\\Cropped\"\n",
    "LABEL_FILE = r\"C:\\Users\\ASIF\\CASME DATASET\\CASME2-coding-20140508.xlsx\"\n",
    "SAVE_DIR = r\"C:\\Users\\ASIF\\CASME DATASET\\casme_loso_runs_20260130_002515\"\n",
    "# ===========================\n",
    "\n",
    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(\"Device:\", DEVICE)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "372fa979-60d4-497f-9e9f-9895844e9dfa",
   "metadata": {},
   "outputs": [],
   "source": [
    "COL_SUBJECT = \"Subject\"\n",
    "COL_FILENAME = \"Filename\"\n",
    "COL_EMOTION = \"Estimated Emotion\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "886cf198-bd3d-4e33-adc3-6e042b105e9c",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CNN_LSTM(nn.Module):\n",
    "    def __init__(self, num_classes):\n",
    "        super().__init__()\n",
    "        cnn = resnet18(weights=ResNet18_Weights.DEFAULT)\n",
    "        for p in cnn.parameters():\n",
    "            p.requires_grad = False\n",
    "\n",
    "        self.cnn = nn.Sequential(*list(cnn.children())[:-1])\n",
    "        self.lstm = nn.LSTM(512, 128, batch_first=True)\n",
    "        self.fc = nn.Linear(128, num_classes)\n",
    "\n",
    "    def forward(self, x):\n",
    "        B, T, C, H, W = x.shape\n",
    "        x = x.view(B*T, C, H, W)\n",
    "        feat = self.cnn(x).flatten(1)\n",
    "        feat = feat.view(B, T, 512)\n",
    "        out, _ = self.lstm(feat)\n",
    "        return self.fc(out[:, -1])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4756e8c9-1b13-44ae-8136-35837b8ab1d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_transform(img_size):\n",
    "    return transforms.Compose([\n",
    "        transforms.ToPILImage(),\n",
    "        transforms.Resize((img_size, img_size)),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize(\n",
    "            mean=[0.485, 0.456, 0.406],\n",
    "            std =[0.229, 0.224, 0.225]\n",
    "        )\n",
    "    ])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "d5330d0f-72e9-491b-a608-00054631d739",
   "metadata": {},
   "outputs": [],
   "source": [
    "def list_frames(path):\n",
    "    return sorted([\n",
    "        f for f in os.listdir(path)\n",
    "        if f.startswith(\"reg_img\") and f.endswith(\".jpg\")\n",
    "    ])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "60880712-8796-4403-8c82-0d0def4b0315",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classes: ['disgust', 'fear', 'happiness', 'others', 'repression', 'sadness', 'surprise']\n",
      "SEQ_LEN: 16 IMG_SIZE: 224\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_excel(LABEL_FILE)\n",
    "\n",
    "# Load one checkpoint to get config\n",
    "ckpt_file = [f for f in os.listdir(SAVE_DIR) if f.startswith(\"best_sub\")][0]\n",
    "ckpt = torch.load(os.path.join(SAVE_DIR, ckpt_file), map_location=DEVICE)\n",
    "\n",
    "emotion_list = ckpt[\"emotion_list\"]\n",
    "emotion_to_id = ckpt[\"emotion_to_id\"]\n",
    "seq_len = ckpt[\"seq_len\"]\n",
    "img_size = ckpt[\"img_size\"]\n",
    "\n",
    "print(\"Classes:\", emotion_list)\n",
    "print(\"SEQ_LEN:\", seq_len, \"IMG_SIZE:\", img_size)\n",
    "\n",
    "transform = get_transform(img_size)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "8f50717c-74f0-49fb-a0db-88f4747511b6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Streaming Subject 1 ...\n",
      "Streaming Subject 2 ...\n",
      "Streaming Subject 3 ...\n",
      "Streaming Subject 4 ...\n",
      "Streaming Subject 5 ...\n",
      "Streaming Subject 6 ...\n",
      "Streaming Subject 7 ...\n",
      "Streaming Subject 8 ...\n",
      "Streaming Subject 9 ...\n",
      "Streaming Subject 10 ...\n",
      "Streaming Subject 11 ...\n",
      "Streaming Subject 12 ...\n",
      "Streaming Subject 13 ...\n",
      "Streaming Subject 14 ...\n",
      "Streaming Subject 15 ...\n",
      "Streaming Subject 16 ...\n",
      "Streaming Subject 17 ...\n",
      "Streaming Subject 18 ...\n",
      "Streaming Subject 19 ...\n",
      "Streaming Subject 20 ...\n",
      "Streaming Subject 21 ...\n",
      "Streaming Subject 22 ...\n",
      "Streaming Subject 23 ...\n",
      "Streaming Subject 24 ...\n",
      "Streaming Subject 25 ...\n",
      "Streaming Subject 26 ...\n"
     ]
    }
   ],
   "source": [
    "y_true, y_pred, latencies = [], [], []\n",
    "\n",
    "for subj in sorted(df[COL_SUBJECT].unique()):\n",
    "    ckpt_path = os.path.join(SAVE_DIR, f\"best_sub{subj:02d}.pt\")\n",
    "    if not os.path.exists(ckpt_path):\n",
    "        continue\n",
    "\n",
    "    model = CNN_LSTM(len(emotion_list)).to(DEVICE)\n",
    "    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)[\"model_state\"])\n",
    "    model.eval()\n",
    "\n",
    "    rows = df[df[COL_SUBJECT] == subj]\n",
    "\n",
    "    print(f\"Streaming Subject {subj} ...\")\n",
    "\n",
    "    for _, r in rows.iterrows():\n",
    "        folder = r[COL_FILENAME]\n",
    "        seq_path = os.path.join(DATA_ROOT, f\"sub{subj:02d}\", folder)\n",
    "        if not os.path.isdir(seq_path):\n",
    "            continue\n",
    "\n",
    "        buffer = []\n",
    "        frames = list_frames(seq_path)\n",
    "\n",
    "        for f in frames:\n",
    "            img = cv2.imread(os.path.join(seq_path, f))\n",
    "            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "            buffer.append(transform(img))\n",
    "            if len(buffer) > seq_len:\n",
    "                buffer.pop(0)\n",
    "\n",
    "            if len(buffer) == seq_len:\n",
    "                seq = torch.stack(buffer).unsqueeze(0).to(DEVICE)\n",
    "                t0 = time.time()\n",
    "                with torch.no_grad():\n",
    "                    pred = torch.argmax(model(seq), dim=1).item()\n",
    "                latencies.append((time.time() - t0) * 1000)\n",
    "\n",
    "        y_true.append(emotion_to_id[r[COL_EMOTION]])\n",
    "        y_pred.append(pred)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "11f07961-6b89-489f-95ca-8c0c32b6b6bb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sequence Accuracy: 0.3843137254901961\n",
      "Avg Latency (ms): 609.985473183046\n",
      "\n",
      "Confusion Matrix:\n",
      " [[20  0  5 29  9  0  0]\n",
      " [ 0  0  0  2  0  0  0]\n",
      " [ 1  0  4 24  3  0  0]\n",
      " [22  0  4 71  1  0  1]\n",
      " [ 5  0  1 18  3  0  0]\n",
      " [ 0  0  0  7  0  0  0]\n",
      " [ 4  0  4 17  0  0  0]]\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "     disgust       0.38      0.32      0.35        63\n",
      "        fear       0.00      0.00      0.00         2\n",
      "   happiness       0.22      0.12      0.16        32\n",
      "      others       0.42      0.72      0.53        99\n",
      "  repression       0.19      0.11      0.14        27\n",
      "     sadness       0.00      0.00      0.00         7\n",
      "    surprise       0.00      0.00      0.00        25\n",
      "\n",
      "    accuracy                           0.38       255\n",
      "   macro avg       0.17      0.18      0.17       255\n",
      "weighted avg       0.31      0.38      0.33       255\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ASIF\\anaconda3\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n",
      "C:\\Users\\ASIF\\anaconda3\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n",
      "C:\\Users\\ASIF\\anaconda3\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n"
     ]
    }
   ],
   "source": [
    "print(\"Sequence Accuracy:\", np.mean(np.array(y_true) == np.array(y_pred)))\n",
    "print(\"Avg Latency (ms):\", np.mean(latencies))\n",
    "\n",
    "print(\"\\nConfusion Matrix:\\n\", confusion_matrix(y_true, y_pred))\n",
    "print(\"\\nClassification Report:\\n\",\n",
    "      classification_report(y_true, y_pred, target_names=emotion_list))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e222277e-7601-4fd2-8545-4f7977d00ff1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
