import configparser

config = configparser.ConfigParser()

config['DEFAULT'] = {'BatchSize': 48,
                     'Regularization': .3}

with open('config.ini', 'w') as configfile:
    config.write(configfile)

print(config['DEFAULT']['BatchSize'])

