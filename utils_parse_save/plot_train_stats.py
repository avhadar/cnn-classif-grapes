import matplotlib.pyplot as plt
from parse_train_log import *


def check_if_string_number(stats_dict):

    # if datatype is string, the numbers should be converted to numeric types
    # (else it might mix up the order of plotted values)

    if type(stats_dict["accuracy"][0]) is str: # str -> needs to be converted
    
        stats_dict["epoch"] = [int(val) for val in stats_dict["epoch"]]
        stats_dict["accuracy"] = [float(val) for val in stats_dict["accuracy"]]
        stats_dict["vali_accuracy"] = [float(val) for val in stats_dict["vali_accuracy"]]
        
        stats_dict["epoch"] = [int(val) for val in stats_dict["epoch"]]
        stats_dict["loss"] = [float(val) for val in stats_dict["loss"]]
        stats_dict["vali_loss"] = [float(val) for val in stats_dict["vali_loss"]]
        
        stats_dict["time"] = [float(val) for val in stats_dict["time"]]
    

def plot_stats_train_vali(stats_dict):
    # Accuracy
    # training data
    plt.scatter(stats_dict["epoch"], stats_dict["accuracy"], color = "blue")
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")
    plt.savefig(r"..\img\training_accuracy_scatter.jpg")
    plt.show()
    
    # validation data
    plt.scatter(stats_dict["epoch"], stats_dict["vali_accuracy"], color = "red")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.savefig(r"..\img\validation_accuracy_scatter.jpg")
    plt.show()
    
    # common plot: training and validation over time (epochs)
    plt.plot(stats_dict["epoch"], stats_dict["accuracy"], "b", label = "training")
    plt.plot(stats_dict["epoch"], stats_dict["vali_accuracy"], "r", label = "validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(r"..\img\accuracy_training_validation.jpg")
    plt.show()
    
    # Loss
    # training data
    plt.scatter(stats_dict["epoch"], stats_dict["loss"], color = "blue")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.savefig(r"..\img\training_loss_scatter.jpg")
    plt.show()
    
    # validation data
    plt.scatter(stats_dict["epoch"], stats_dict["vali_loss"], color = "red")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.savefig(r"..\img\validation_loss_scatter.jpg")
    plt.show()
    
    # common plot: training and validation over time (epochs)
    plt.plot(stats_dict["epoch"], stats_dict["loss"], "b", label = "training")
    plt.plot(stats_dict["epoch"], stats_dict["vali_loss"], "r", label = "validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(r"..\img\loss_training_validation.jpg")
    plt.show()
    
    # Time per epoch
    plt.plot(stats_dict["epoch"], stats_dict["time"], "g")
    plt.xlabel("Epoch")
    plt.ylabel("Time")
    plt.savefig(r"..\img\time_per_epoch.jpg")
    plt.show()


def generate_overall_stats(stats_dict, outfile = "overall_stats.txt"): 
    
    # Generate statistics: mean, min, max, start, end
    
    # open stats output file
    out_stats = open(outfile, "w")
    
    # Accuracy
    # training
    print("\nTraining data - accuracy:")
    print("Min: ", min(stats_dict["accuracy"]))
    print("Max: ", max(stats_dict["accuracy"]))
    print("Mean: ", sum(stats_dict["accuracy"]) / len(stats_dict["accuracy"]) )
    print("Start value: ", stats_dict["accuracy"][0])
    print("End value: ", stats_dict["accuracy"][len(stats_dict["accuracy"]) - 1])
    
    out_stats.write("Training data - accuracy:")
    out_stats.write("\nMin: " + str(min(stats_dict["accuracy"])) )
    out_stats.write("\nMax: " + str(max(stats_dict["accuracy"])) )
    out_stats.write("\nMean: " + str(sum(stats_dict["accuracy"]) / len(stats_dict["accuracy"])) )
    out_stats.write("\nStart value: " + str(stats_dict["accuracy"][0]) )
    out_stats.write("\nEnd value: " + str(stats_dict["accuracy"][len(stats_dict["accuracy"]) - 1]) )
    out_stats.write("\n")
    
    # validation
    print("\nValidation data - accuracy:")
    print("Min: ", min(stats_dict["vali_accuracy"]))
    print("Max: ", max(stats_dict["vali_accuracy"]))
    print("Mean: ", sum(stats_dict["vali_accuracy"]) / len(stats_dict["vali_accuracy"]) )
    print("Start value: ", stats_dict["vali_accuracy"][0])
    print("End value: ", stats_dict["vali_accuracy"][len(stats_dict["vali_accuracy"]) - 1])
    
    out_stats.write("\nValidation data - accuracy:")
    out_stats.write("\nMin: " + str(min(stats_dict["vali_accuracy"])) )
    out_stats.write("\nMax: " + str(max(stats_dict["vali_accuracy"])) )
    out_stats.write("\nMean: " + str(sum(stats_dict["vali_accuracy"]) / len(stats_dict["vali_accuracy"])) )
    out_stats.write("\nStart value: " + str(stats_dict["vali_accuracy"][0]) )
    out_stats.write("\nEnd value: " + str(stats_dict["vali_accuracy"][len(stats_dict["vali_accuracy"]) - 1]) )
    out_stats.write("\n")
    
    # Loss
    # training
    print("\nTraining data - loss:")
    print("Min: ", min(stats_dict["loss"]))
    print("Max: ", max(stats_dict["loss"]))
    print("Mean: ", sum(stats_dict["loss"]) / len(stats_dict["loss"]) )
    print("Start value: ", stats_dict["loss"][0])
    print("End value: ", stats_dict["loss"][len(stats_dict["loss"]) - 1])
    
    out_stats.write("\nTraining data - loss:")
    out_stats.write("\nMin: " + str(min(stats_dict["loss"])) )
    out_stats.write("\nMax: " + str(max(stats_dict["loss"])) )
    out_stats.write("\nMean: " + str(sum(stats_dict["loss"]) / len(stats_dict["loss"])) )
    out_stats.write("\nStart value: " + str(stats_dict["loss"][0]) )
    out_stats.write("\nEnd value: " + str(stats_dict["loss"][len(stats_dict["loss"]) - 1]) )
    out_stats.write("\n")
    
    # validation
    print("\nValidation data - loss:")
    print("Min: ", min(stats_dict["vali_loss"]))
    print("Max: ", max(stats_dict["vali_loss"]))
    print("Mean: ", sum(stats_dict["vali_loss"]) / len(stats_dict["vali_loss"]) )
    print("Start value: ", stats_dict["vali_loss"][0])
    print("End value: ", stats_dict["vali_loss"][len(stats_dict["vali_loss"]) - 1])
    
    out_stats.write("\nValidation data - loss:")
    out_stats.write("\nMin: " + str(min(stats_dict["vali_loss"])) )
    out_stats.write("\nMax: " + str(max(stats_dict["vali_loss"])) )
    out_stats.write("\nMean: " + str(sum(stats_dict["vali_loss"]) / len(stats_dict["vali_loss"])) )
    out_stats.write("\nStart value: " + str(stats_dict["vali_loss"][0]) )
    out_stats.write("\nEnd value: " + str(stats_dict["vali_loss"][len(stats_dict["vali_loss"]) - 1]))
    out_stats.write("\n")
    
    # Time
    print("\nTime per epoch (s):")
    print("Min: ", min(stats_dict["time"]))
    print("Max: ", max(stats_dict["time"]))
    print("Mean: ", sum(stats_dict["time"]) / len(stats_dict["time"]) )
    print("Start value: ", stats_dict["time"][0])
    print("End value: ", stats_dict["time"][len(stats_dict["time"]) - 1])
    
    print("\nTotal time (all epochs): ", sum(stats_dict["time"]))
    
    out_stats.write("\nTime per epoch (s):")
    out_stats.write("\nMin: " + str(min(stats_dict["time"])) )
    out_stats.write("\nMax: " + str(max(stats_dict["time"])) )
    out_stats.write("\nMean: " + str(sum(stats_dict["time"]) / len(stats_dict["time"])) )
    out_stats.write("\nStart value: " + str(stats_dict["time"][0]) )
    out_stats.write("\nEnd value: " + str(stats_dict["time"][len(stats_dict["time"]) - 1]) )

    out_stats.write("\nTotal time (all epochs): " + str(sum(stats_dict["time"])) )
    
    # close stats output file (all data has been written)
    out_stats.close()


# path to output of model training
train_log = r".\..\logs\model_training_log.txt"

# parse the training log
parse_log(train_log)

# convert numbers stored as strings -> back to numbers
check_if_string_number(epoch_stats)

# plot the epoch statistics
plot_stats_train_vali(epoch_stats)

# generate summary stats (over all epochs)
generate_overall_stats(epoch_stats)