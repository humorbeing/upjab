from omegaconf import OmegaConf

from omegaconf import OmegaConf

# Create an empty OmegaConf dictionary
conf = OmegaConf.create()

args = {
    'hi': 1,
    'hello': 2
}
conf = OmegaConf.create(args)

config_path = 'data/configs/test.yaml'
conf = OmegaConf.load(config_path)
print(conf)
print(conf.lr)


def test_(lr, **kwargs):
    print(lr)
    if kwargs:
        print(kwargs)
    else:
        print('no kwargs')

test_(**conf)


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