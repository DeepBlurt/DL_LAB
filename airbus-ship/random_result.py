import os
import random
import csv
import pandas as pd


test_data = "/media/ax/文件/contest/airbus-ship/test"
image_list = os.listdir(test_data)
print(len(image_list))
sample_submission = "./sample_submission.csv"
df = pd.read_csv(sample_submission)

images = df["ImageId"]
print(len(df["ImageId"]))

with open("./submit.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file, dialect="excel")
    writer.writerow(["ImageId", "EncodedPixels"])
    for image in images:
        masks = random.randint(0, 200)
        pair = ""
        for mask in range(masks):
            start = random.randint(2000, 500000)
            length = random.randint(50, 300)
            pair = pair + str(start) + " " + str(length) + " "
        writer.writerow([image, pair])
