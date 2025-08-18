from omegaconf import OmegaConf

config_path = 'data/configs/test.yaml'
conf = OmegaConf.load(config_path)
print(conf)
print(conf.lr)

import sys

sys.argv = ['your_program.py', 'server.port=8080', 'log.level=INFO', 'database.host=localhost']
conf1 = OmegaConf.from_cli()
print(conf1)

sys.argv = ['config_read_module.py', 'server.port=8080', 'log.level=INFO', 'database.host=localhost']
conf1 = OmegaConf.from_cli()
print(conf1)

sys.argv = ['server.port=8080', 'log.level=INFO', 'database.host=localhost']
conf1 = OmegaConf.from_cli()
print(conf1)

print('done')