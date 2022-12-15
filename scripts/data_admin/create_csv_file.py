import csv


directory = os.fsencode("gen_images/24-Sep__21-03__array-1/")
header = ['image_path', 'object', 'location', 'time_of_day']
    
with open("gen_images/csv_files/24-Sep__21-03.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"): 
            
            # obtain labels in format [object, location, time]
            labels = filename[:-4]
            labels = labels.split('__')[-1]
            labels = labels.split("-")
            
            data = [filename, labels[0], labels[1], labels[2]]
            writer.writerow(data)