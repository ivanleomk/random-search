import csv

# Open the CSV file
with open("./data/trials.csv", mode="r") as file:
    # Create a CSV reader
    csv_reader = csv.DictReader(file)
    headers = next(csv_reader)
    excluded_params = set(
        ["metric_precision", "", "metric_recall", "metric_accuracy", "metric_AUC"]
    )
    params = {header: set() for header in headers if header not in excluded_params}

    count = 0
    for row in csv_reader:
        for metric in row:
            if metric in excluded_params:
                continue
            params[metric].add(row[metric])
        count += 1

    print(count)

    for param in params:
        param_items = " , ".join(list(params[param]))
        print(f"{param}=[{param_items}]")
