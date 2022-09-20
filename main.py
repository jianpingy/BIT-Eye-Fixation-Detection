#Main

debug = False

raw_data = pd.read_csv('CW_Mar_10_ET1_O_1CMD.txt', sep='\t')
T = raw_data['Timestamp']
XL = raw_data['GazepointX (L)']
YL = raw_data['GazepointY (L)']
XR = raw_data['GazepointX (R)']
YR = raw_data['GazepointY (R)']

(T,Z,(ind_fix,ind_sac,ind_blink)) = BIT(T,XL,YL,XR,YR,alpha=0.25,x_min=0,x_max=1281,y_min=0,y_max=769,debug=debug)
indexes, num_fix = Fixation_Classification(Z.shape[0], ind_fix, ind_sac, ind_blink)

matlab_results = pd.read_csv('result.txt_fix_stamps.txt', sep='\t')
matlab_labels = matlab_results['label']
        
Equal = []
for i in range(len(matlab_labels)):
    if int(indexes[i].item()) == int(matlab_labels[i]):
        Equal.append('True')
    else:
        Equal.append('False')

if debug:
    textfile_name = 'data_debug.txt'
else:
    textfile_name = 'data.txt'
    
with open(textfile_name, 'w') as f:
    f.write('Individual\tTime\tx_left\ty_left\tx_right\ty_right\tdistance\tfix_label_my_version\tfix_label_matlab\n')
    for i,line in enumerate(Z):
        f.write(str(0)+'\t')
        f.write(str(T[i].item())+'\t'+str(line[0].item())+'\t'+str(line[1].item())+'\t'+str(line[2].item())+'\t'+str(line[3].item())+'\t'
                +str(1.0)+'\t'+str(int(indexes[i].item()))+'\t'+str(matlab_labels[i])+'\t'+Equal[i]+'\n')
