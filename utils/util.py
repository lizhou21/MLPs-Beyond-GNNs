import json
import pickle
import random


def read_json(file):
    data = []
    with open(file, "r") as json_file:
        for line in json_file:
            data.append(json.loads(line.strip()))
    return data


def write_json(file, data):
    with open(file, "w") as json_file:
        for line in data:
            json_file.write(json.dumps(line) + "\n")


def read_pickle(file):
    with open(file, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data


def write_pickle(file, data):
    with open(file, "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def read_json_random(file):
    count = 0
    data = []
    with open(file, 'r') as json_file:
        for line in json_file:
            try:
                person = json.loads(line.strip())
                random_edge = []
                for i, edge in enumerate(person['edges']):
                    numbers = list(range(1, len(person['edges'])))
                    if edge[1] in numbers:
                        numbers.remove(edge[1])  # 移除x
                    start_id = random.choice(numbers) - 1
                    random_edge.append([start_id, edge[1]])
                person['edges'] = random_edge
                data.append(person)
            except:
                person = json.loads(line.strip())
                data.append(person)
                count = count + 1

    print(file, "fail", count)
    return data