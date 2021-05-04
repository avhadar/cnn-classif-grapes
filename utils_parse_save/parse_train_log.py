import re

logfile = r".\..\logs\model_training_log.txt"

regex_dict = { 'epoch'         : re.compile("(\d+)/(\d+)"),
               'time'          : re.compile("(\d+)s\s+"),
               'time_per_step' : re.compile("(\d+)([ms]+)/step"),
               'loss'          : re.compile("loss:\s+(\d+\.\d+)\s+"),
               'accuracy'      : re.compile("accuracy:\s+(\d+\.\d+)\s+"),
               'vali_loss'     : re.compile("val_loss:\s+(\d+\.\d+)\s+"),
               'vali_accuracy' : re.compile("val_accuracy:\s+(\d+\.\d+)\s*")
             }

epoch_stats = { 'epoch'         : [],
               'time'          : [],
               'time_per_step' : [],
               'loss'          : [],
               'accuracy'      : [],
               'vali_loss'     : [],
               'vali_accuracy' : [],
             }

def parse_log(logfile):

    with open(logfile, "r") as train_log:
        print("Opened training log. Processing...")
        
        while True:
            line = train_log.readline()
            
            if not line:
                break
                
            if "Epoch " in line:
                # get epoch count
                # get total epoch count
                ep_count = regex_dict['epoch'].search(line).group(1)
                # print(ep_count)
                epoch_stats['epoch'].append(ep_count)
                
                # read next line
                ep_details = train_log.readline()
                # get time(s)
                ep_time = regex_dict['time'].search(ep_details).group(1)
                # print(ep_time)
                epoch_stats['time'].append(ep_time)
                
                # get step duration ((m)s/step)
                ep_t_and_unit = regex_dict['time_per_step'].search(ep_details)
                # print(ep_t_and_unit)
                ep_t_step = ep_t_and_unit.group(1)
                ep_t_unit = ep_t_and_unit.group(2)
                # print(ep_t_step)
                # print(ep_t_unit)
                epoch_stats['time_per_step'].append((ep_t_step, ep_t_unit))
                
                # get loss
                ep_loss = regex_dict['loss'].search(ep_details).group(1)
                # print(ep_loss)
                epoch_stats['loss'].append(ep_loss)
                
                # get accuracy
                ep_acc = regex_dict['accuracy'].search(ep_details).group(1)
                # print(ep_acc)
                epoch_stats['accuracy'].append(ep_acc)
                
                # get validation loss
                ep_vali_loss = regex_dict['vali_loss'].search(ep_details).group(1)
                # print(ep_vali_loss)
                epoch_stats['vali_loss'].append(ep_vali_loss)
                
                # get validation accuracy
                ep_vali_acc = regex_dict['vali_accuracy'].search(ep_details).group(1)
                # print(ep_vali_acc)
                epoch_stats['vali_accuracy'].append(ep_vali_acc)
    
    print("Log processing finished")
        
if __name__ == "__main__":
    parse_log(logfile)