import os.path
import pickle
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
config_txt_path = dir_path + '/list_of_runs.p'


### DEFAULTS CHANGE IF YOU WANT TO MAKE RUNS QUICKLY THROUGH THE MENU
default_save_params = {'good_path':'/home/eddann/Monash/Edwards/Image_Data/good/res_60' ,
                       'bad_path': '/home/eddann/Monash/Edwards/Image_Data/bad/res_60',
                       'save_model_path': '/home/eddann/Monash/Edwards/Model',
                       'bench_path': '/home/eddann/Monash/Edwards/benching.p'}

default_training_params = {'box_size': '240',
                           'batch_size': '100',
                           'test_size': '500',
                           'num_iters': '5000'}

default_model_params = {'filter_dim1': '5',
                        'num_filters1': '11',
                        'filter_dim2': '5',
                        'num_filters2':'25',
                        'full_con_size1': '1000',
                        'full_con_size2': '900'}



### ADD RUNS HERE IF YOU PREFER AND THEN LAUNCH MENU AND PRESS M
def add_manually(run_params):

    run_params.append({

    'good_path':        '/home/serv/Downloads/Images/good',
    'bad_path':         '/home/serv/Downloads/Images/BAD240FULL',
    'save_model_path':  '/home/serv/Downloads/Images',
    'bench_path':       '/home/serv/Downloads/Images',

    'box_size':         "240",
    'batch_size':       '100',
    'test_size':        '500',
    'num_iters':        '5000',

    'filter_dim1':      '5',
    'num_filters1':     '11',
    'filter_dim2':      '5',
    'num_filters2':     '25',
    'filter_dim3':      '7',
    'num_filters3':     '41',
    'filter_dim4':      '11',
    'num_filters4':     '25',

    'full_con_size1':   '1000',
    'full_con_size2':   '900'

    })

    return run_params






def input_loop(path):


    exit_flag = False
    run_params = pickle.load(open(path, "rb"))


    options = ['good_path','bad_path','save_model_path','batch_size','test_size','num_iters','filter']
    extra_option = ['test_batch_size']
    while exit_flag == False:
        for i in range(50):
            print('')
        print("Welcome to the Parameter menu, you can see all runs, delete or modify exist runs")
        print("menu: Select an option based on number (ie 1)")
        print("1) Show all runs          2) Create new run")
        print("3) Delete all runs        4) Modify an existing run")
        print("5) Disable a run          6) Enable a run")
        print('9) Save and Exit          0) Exit without Saving')
        print('To add Manually press M')
        for i in range(8):
            print('')
        user_input = input('Enter an integer: ')
        for i in range(50):
            print('')
        i = 1

        if user_input == 'M':
            run_params = add_manually(run_params)
        
        if user_input == '1':
            if run_params == []:
                print("No runs created!")
            else:
                print("Run: " + str(i))
                i += 1
                for d in run_params:
                    print()
                    for k, v in d.items():
                        print(k, '-->', v)
            input('Press Enter to Return to the menu')

        if user_input == '2':
            new_run = {}

            ##################   Path Params     ##################

            print('Use default params for Saving and Loading Data?')
            print('')
            for key, value in default_save_params.items():
                print(key + ' = ' + str(value))
            print('')
            default_save_input = input('Y/N?  ->  ')
            if default_save_input in ['y','Y']:
                for key,value in default_save_params.items():
                    new_run[key] = value
            else:
                new_run['good_path'] = input('Good Path: ')
                new_run['bad_path'] = input('Bad Path: ')
                new_run['save_model_path'] = input('Save Model Path: ')
                new_run['bench_path'] = input('Benchmarking Stats Path')
            for i in range(50):
                print('')

            ##################   Training Params     ##################

            print('Use default params for Training the Model?')
            print('')
            for key, value in default_training_params.items():
                print(key + ' = ' + str(value))
            print('')

            default_training_input = input('Y/N?  ->  ')
            if default_training_input in ['y','Y']:
                for key,value in default_training_params.items():
                    new_run[key] = value
            else:
                new_run['batch_size'] = input('Training Batch Size: ')
                new_run['test_size'] = input('Test Batch Size: ')
                new_run['num_iters'] = input('Number of Iterations (Training Runs for this model): ')
            for i in range(50):
                print('')

            ##################   Model Params     ##################
            print('Use default params for Training the Model?')
            print('')
            for key, value in default_model_params.items():
                print(key + ' = ' + str(value))
            print('')
            default_model_input = input('Y/N?  ->  ')
            if default_model_input in ['y','Y']:
                for key, value in default_model_params.items():
                    new_run[key] = value
            else:

                print('Params for Conv Net 1')
                print('')
                new_run['filter_dim1'] = input('Filter Size: ')
                new_run['num_filters1'] = input('Number of Filters: ')
                print('Params for Conv Net 2')
                print('')
                new_run['filter_dim2'] = input('Filter Size: ')
                new_run['num_filters2'] = input('Number of Filters: ')
                print('Params for Fully Connected Layers')
                new_run['full_con_size1'] = input('Size of FCL 1: ')
                new_run['full_con_size1'] = input('Size of FCL 2: ')
            for i in range(50):
                print('')
            print("Do you wish to create a run with these parameters?")
            for key, value in new_run.items():
                print(key + ' = ' + str(value))
            create_input = input('Y/N?  ->  ')
            if create_input in ['y','Y']:
                run_params.append(new_run)

        if user_input == '3':
            file = pickle.dump([], open(config_txt_path, "wb"))
            run_params = []
        if user_input == '0':
            exit_flag = True
        if user_input == '9':
            pickle.dump(run_params,open('list_of_runs.p',"wb"))
            exit_flag = True



########################################## MAIN  #################################################




if os.path.exists(config_txt_path):
    print('Found Parameters, loading params')
    input_loop(config_txt_path)

else:
    print('No Parameters found, would you like to create a new param config file?')
    user_input = input('Y/N?  ->  ')
    if user_input in ['y','Y']:
        print('Config File Created at ' + dir_path)
        file = pickle.dump( [], open(config_txt_path, "wb" ) )
        input_loop(config_txt_path)
    else:
        print("Well then go find some params for me to run!")
        exit(0)







