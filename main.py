#Main

raw_data = pd.read_csv('data_file_name.txt', sep='\t')
T = raw_data['Timestamp']
XL = raw_data['GazepointX (L)']
YL = raw_data['GazepointY (L)']
XR = raw_data['GazepointX (R)']
YR = raw_data['GazepointY (R)']

(T,(ind_fix,ind_sac,ind_blink)) = BIT(T,XL,YL,XR,YR,alpha=0.25,x_min=0,x_max=1281,y_min=0,y_max=769)
