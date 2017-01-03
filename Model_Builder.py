import os.path
import pickle
import os
import sys
import draw_convert
import CNNcls
import Manual_Model_Creation
dir_path = os.path.dirname(os.path.realpath(__file__))
config_txt_path = dir_path + '/list_of_runs.p'







auto_load = True







### DEFAULTS CHANGE IF YOU WANT TO MAKE RUNS QUICKLY THROUGH THE MENU

DEFAULT_IMAGE_LOCATION = str(dir_path)      +'/Images'     #Change to an absolute path if Image Directory is not located within Build_Folder
DEFAULT_EXPORT_LOCATION = str(dir_path)




def create_model(name):

    new_model = {}
    new_model['name'] = name


    #Configuration of Paths and stuff
    print("Use Default Export and Image Directory Locations?")
    u_input = input('Y/N?  ->  ')
    if u_input in ['y','Y','yes']:
        print('')
        print('')
        new_model['Build_Directory']    = DEFAULT_EXPORT_LOCATION
        new_model['Image_Dataset_Path'] = DEFAULT_IMAGE_LOCATION
        print('Using Build Directory: '+ DEFAULT_EXPORT_LOCATION)
        print('Using Image Directory: '+ DEFAULT_IMAGE_LOCATION)
        input("Press Enter to Continue")
    else:
        new_model['Build_Directory']    = input("Location to save and export Models:  ")
        new_model['Image_Dataset_Path'] = input("Location of Image Dataset:   ")


        pass

    #Global vars of the model
    for i in range(50):
        print('')

    print("Global Parameters:")
    print('')

    new_model['num_iters']  = input("Number of Training Iterations: ")
    new_model['box_size']   = input("Box Size or resolution of input images: ")
    new_model['batch_size'] = input("Training Batch Size: ")
    new_model['test_size']  = input("Testing Batch Size: ")

    ## LAYER CREATION ###
    for i in range(50):
        print('')
    print("Layer Creation")
    print('')
    new_model['layers'] = create_layers(new_model)
    print('Are these parameters correct for you model?')
    for k, v in new_model.items():
        print(k, '-->', v)
    if input("(Y/n) --> ") in ['y','Y','yes']:
        return new_model
    else:
        print("Starting Again")
        return create_model(name)

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
            
def Train_Models(list_of_runs):
    print('Loading Params')
    run_params = list_of_runs
    for items in run_params:
        model = CNNcls.CNNcls(run_params=items)
        model.train_model()

def input_loop(path):
    exit_flag = False
    model_list = pickle.load(open(path, "rb"))

    options = ['good_path', 'bad_path', 'save_model_path', 'batch_size', 'test_size', 'num_iters', 'filter']
    extra_option = ['test_batch_size']
    while exit_flag == False:
        for i in range(50):
            print('')
        print("Welcome to the Model Builder, Here you can create new models")
        print("menu: Select an option based on number (ie 1)")
        print("1) Show all runs          2) Create new model")
        print("3) Delete all runs        4) Modify an existing run")
        print("5) Add Models to training Queue")
        print("6) Train Models in Queue")
        print('9) Save and Exit          0) Exit without Saving')
        print('To add Manually press M')
        for i in range(8):
            print('')
        
        
        user_input = input('Enter an integer: ')
        
        for i in range(50):
            print('')
        i = 1

       
        if user_input == 'M':
            model_list = Manual_Model_Creation.add_manually(model_list,dir_path)
            
        if user_input == '6':
            Train_Models(model_list)

        if user_input == '1':
            if model_list == []:
                print("No runs created!")
            else:
                print("######################### Run: " + str(i)+ ' ########################')
                i += 1
                for d in model_list:
                    print()
                    for k, v in d.items():
                        print(k, '-->', v)
            graphical = input("Press 'S' to see the models graphically, or press Enter to return to menu: ")
            if graphical in ['S','s']:
              print("Launching Ipython to Show Graphs")          
              for model in model_list:             
                 draw_convert.draw_from_model_params(model)           

        if user_input == '2':
            name = input("Name of model:   ")
            model_list.append(create_model(name))

        if user_input == '3':
            file = pickle.dump([], open(config_txt_path, "wb"))
            model_list = []
            
        if user_input == '0':
            exit_flag = True
            
        if user_input == '9':
            pickle.dump(model_list, open('list_of_runs.p', "wb"))
            exit_flag = True


########################################## MAIN  #################################################


#Make sure we can find images
if auto_load == True:
    print("Building from manual model creation")
    file = pickle.dump([], open(config_txt_path, "wb"))
    model_list = []
    model_list = Manual_Model_Creation.add_manually(model_list, dir_path)
    pickle.dump(model_list, open('list_of_runs.p', "wb"))
    Train_Models(model_list)


if os.path.exists(config_txt_path):
    print('Found Settings File, Initializing')
    input_loop(config_txt_path)

else:
    print('No Settings File Found, Would you like to create a new Config File?')
    user_input = input('Y/N?  ->  ')
    if user_input in ['y', 'Y']:
        print('Config File Created at ' + dir_path)
        print('Using Default Image location:' )
        if not os.path.exists(DEFAULT_IMAGE_LOCATION):
            print("IMAGE DIRECTORY NOT FOUND")
            downl = input("Download Image Dataset and Place in Build Directory Now? (y/n) -->   ")
            if downl in ['y','Y','yes']:
                ########################
                ########################
                ###########DL###########
                ###########CODE#########
                ###########TBF##########
                pass
            else:
                print("Change The Image Location at the Top of this Script to correct path")
                print('')
                input("Press Enter to exit")
                exit(0)

        print('Using Default Exporting Directory Settings')
        file = pickle.dump([], open(config_txt_path, "wb"))
        input_loop(config_txt_path)
    else:
        print("Well then go find some params for me to run!")
        exit(0)







def get_build_directory():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path
