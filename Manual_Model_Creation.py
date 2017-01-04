



#### Add multiple models using this function

def add_manually(run_params,dir_path):
    #To create multiple models cut the code bewlo
    #M1
    #COPY FROM HERE

    #M1
    new_model = {

        # New Directory notation
        'Image_Dataset_Path': str(dir_path) + "/Images",
        'Build_Directory': str(dir_path),
        'name': 'test1',
        'box_size': "60",
        'batch_size': '100',
        'test_size': '500',
        'num_iters': '5000',
        'MS': 'True',
        'dropout': 'False'

    }
    con_layers = [(5,9),(5,25)]
    fcl_layers = [1000,600]
    print("adding model param " + str(new_model))
    new_model['layers'] = create_layers(con_params=con_layers,fcl_params=fcl_layers)
    run_params.append(new_model)
    # TO HERE

    #M2
    new_model = {

        # New Directory notation
        'Image_Dataset_Path': str(dir_path) + "/Images",
        'Build_Directory': str(dir_path),
        'name': 'test2',
        'box_size': "240",
        'batch_size': '100',
        'test_size': '500',
        'num_iters': '5000',
        'MS': 'True',
        'dropout': 'False'
    }
    con_layers = [(5,9),(5,25)]
    fcl_layers = [1000,600]
    print("adding model param " + str(new_model))
    new_model['layers'] = create_layers(con_params=con_layers,fcl_params=fcl_layers)
    run_params.append(new_model)

#   #M3
    new_model = {

        # New Directory notation
        'Image_Dataset_Path': str(dir_path) + "/Images",
        'Build_Directory': str(dir_path),
        'name': 'test3',
        'box_size': "120",
        'batch_size': '100',
        'test_size': '500',
        'num_iters': '5000',
        'MS': 'True',
        'dropout': 'False'
    }
    con_layers = [(5,9),(5,25),(11,11)]
    fcl_layers = [1000,600]
    print("adding model param " + str(new_model))
    new_model['layers'] = create_layers(con_params=con_layers,fcl_params=fcl_layers)
    run_params.append(new_model)

    
    #End
    input("Press Enter to continue")
    return run_params



def create_layers(con_params=[],fcl_params=[]):

    layers = {}
    con_layers = []
    fcl_layers = []


    # Layers can be Created via Without the UI by running Create Layers and giving it
    # This would be used if you want to create a hundred fully connect layers or something

    #Con Layer Loop
    c = 0
    if not con_params == []:
        new_con_layer = {}
        num_con_layers = len(con_params)
        while c < num_con_layers:
            new_con_layer = {}
            print("adding c_layer " + str(con_params[c]))
            new_con_layer['filter_dim']  = con_params[c][0]
            new_con_layer['num_filters'] = con_params[c][1]
            con_layers.append(new_con_layer)
            c += 1
    else:
        num_con_layers = int(input("Number of Convoluted Layers: "))
        print('')
        while c < num_con_layers:
            new_con_layer = {}
            print('Params for Convolutional Layer ' + str(c+1))
            print('')
            new_con_layer['filter_dim'] = input('Filter Size (width of kernel): ')
            new_con_layer['num_filters'] = input('Number of Filters: ')
            con_layers.append(new_con_layer)
            c+= 1

    #Fcl Layer loop
    f = 0
    if not fcl_params == []:
        num_fcl_layers = len(fcl_params)
        while f < num_fcl_layers:
            new_fcl_layer = {}
            new_fcl_layer['full_con_size'] = fcl_params[f]
            fcl_layers.append(new_fcl_layer)
            f+= 1
    else:
        num_fcl_layers = int(input("Number of Fully Connected Layers: "))
        print('')
        print("Specify Sizes of Fully Connected Layers")
        while f < num_fcl_layers:
            new_fcl_layer = {}
            stringy = 'Size of Fully Connect Layer ' + str(f+1)
            print('')
            new_fcl_layer['full_con_size'] = input(stringy)
            fcl_layers.append(new_fcl_layer)
            f+= 1
    layers['conv_layers'] = con_layers
    layers['fcl_layers']  = fcl_layers
    return layers


# The Default model, adds a single model to the list of run_params
### To be done later, refactor run_params to list_model_params

def get_default_model_settings(run_params, dir_path):
    new_model = {

        # New Directory notation
        'Image_Dataset_Path': str(dir_path) + "/Images",
        'Build_Directory': str(dir_path),
        'name': 'test',
        'box_size': "240",
        'batch_size': '100',
        'test_size': '500',
        'num_iters': '5000',
        'MS': 'True'

    }
    con_layers = [(5, 11), (12, 25)]
    fcl_layers = [1000, 600]
    print("adding model param " + str(new_model))
    new_model['layers'] = create_layers(con_params=con_layers, fcl_params=fcl_layers)
    run_params.append(new_model)

    return run_params


def get_build_directory():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path
