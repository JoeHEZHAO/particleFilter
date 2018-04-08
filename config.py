## Config for Particle Filter object
'''
Data: 2018.02.24
Name: He Zhao
Inistitue: Yok Unidersity, Vision Lab
Advisor: Richard Wildes

'''

para = {}
para['particle_mean'] = 0.5
para['particle_sigma'] = 0.5
para['highlight_period'] = [63, 71]
para['process_noise_mean'] = 1
para['process_noise_sigma'] = 0
para['errorPrior_mean'] = 0
para['errorPrior_sigma'] = 1
para['numOfParticle'] = 500

data_config = {}
data_config['root_dir'] = '/home/zhufl/Data/soe/'
data_config['data_name'] = 'video1.mat'
data_config['highlight'] = [63, 71]

if __name__ == '__main__':
	pass
