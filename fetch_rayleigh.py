from cl61.fetch.raw import fetch_raw, fetch_model
import os

for date in [
    "2023-01-01",
    "2023-01-02",
    "2023-03-31",
    "2023-04-01",
    "2023-05-13",
    "2023-05-14",
    "2023-07-11",
    "2023-07-12",
    "2023-09-17",
    "2023-09-18",
    "2024-01-01",
    "2024-01-02",
    "2024-02-08",
    "2024-02-09",
    "2024-03-05",
    "2024-03-06",
    "2024-05-04",
    "2024-05-05",
    "2024-06-25",
    "2024-06-26",
    "2024-08-07",
    "2024-08-08",
    "2024-10-28",
    "2024-10-29",
    "2024-12-30",
    "2024-12-31",
]:
    directory = "/media/viet/CL61/studycase/hyytiala/" + date.replace("-", "")

    # Check if directory exists, otherwise create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
        os.makedirs(directory + "/weather/")
    else:
        print(f"Directory already exists: {directory}")
        continue
    fetch_raw(
        "hyytiala",
        date,
        date,
        directory + "/",
    )
    fetch_model(
        "hyytiala",
        date,
        date,
        directory + "/weather/",
    )
