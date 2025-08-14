from datetime import datetime
import json
class Logger:
    def __init__(self):
        self.log_path = \
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S.json")}.json'
        # 新建log_name文件
        with open(self.log_path, 'w') as f:
            json.dump({}, f, indent=4)

    def write(self, context):
        # 读取log_name文件
        with open(self.log_path, 'r') as f:
            data = json.load(f)
        # 将context加入log文件
        data.update(context)
        with open(self.log_path, 'w') as f:
            json.dump(data, f, indent=4)
