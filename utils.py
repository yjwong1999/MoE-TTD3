import json

with open("data_file.json", "r") as read_file:
    data = json.load(read_file)

print(data)


import json

data = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

json_string = json.dumps(data)

with open("data_file.json", "w") as write_file:
    write_file.write(json_string)
