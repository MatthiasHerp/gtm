import os
import mlflow
import matplotlib.pyplot as plt

def log_mlflow_plot(fig,file_name,temporary_storage_directory="./temp/",type="plt"):

    #the_wd = os.getcwd()

    if type == "plt":
        fig.savefig(temporary_storage_directory +  "/" + file_name)
        plt.close(fig) # need to close explicitly because matplotlib figures are opened and consume a lot of memory
    elif type == "plotly":
        fig.write_html(temporary_storage_directory +  "/" + file_name)
        #html figures dont need explicity closing because once they are generated they are like a file

    mlflow.log_artifact(temporary_storage_directory +  "/" + file_name) #"./" the_wd
    

def create_temp_folder(temp_folder):
    # create a folder called temp
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
       
    
def clear_temp_folder(temp_folder):
    # delete all files in the temp folder
    for file in os.listdir(temp_folder):
        os.remove(temp_folder + "/" + file)
    # delete the temp folder
    os.rmdir(temp_folder)
    