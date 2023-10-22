import os, json

def update_results(json_filename, project_name, reward, ave_ssr=0):
    if not os.path.isfile(json_filename):
        # create data
        data = {}
    else:
        # read json file
        with open(json_filename, "r") as read_file:
            data = json.load(read_file)

    # add new data
    data[project_name] = {'reward': reward,\
                          'ave_ssr': ave_ssr
                         }

    # write data
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
