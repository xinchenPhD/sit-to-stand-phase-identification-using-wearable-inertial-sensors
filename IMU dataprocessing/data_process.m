clear;
close all;
clc;
addpath('quaternion_library');  
addpath('ximu_matlab_library'); 
XSENS_path = 'C:\Users\Administrator\Desktop\cx\XSENS data viewV1.0 20170822 (1)\XSENS data viewV1.0 20170822\';
XSENS_name = '*.txt';   
XSENSPath = dir([XSENS_path, XSENS_name]);
Fix_Path = 'C:\Users\Administrator\Desktop\cx\XSENS data viewV1.0 20170822 (1)\XSENS data viewV1.0 20170822\';

Length = length(XSENSPath );    
for i = 1:Length       
    SamplePeriod = 1/100;
    temp = importdata(strcat(XSENS_path ,XSENSPath (i).name), '\t', 6); 
    XSENSDATA = temp.data;
    raw_data = XSENSDATA(:,2:11);
    row=size(XSENSDATA,1);
    XSENS = struct;
    XSENS.time = ((0:row-1)*SamplePeriod)';
    %Original 3-axis acc and gyro 
    Accelerometer = XSENSDATA(:,2:4);       %m/s2
    Accelerometer_X= XSENSDATA(:,2);   
    Accelerometer_Y = XSENSDATA(:,3); 
    Accelerometer_Z = XSENSDATA(:,4); 

    Gyroscope = XSENSDATA(:,5:7);       % rad
    Gyroscope_angle = XSENSDATA(:,5:7)*180/pi; 
    Gyroscope_X = XSENSDATA(:,5);
    Gyroscope_Y = XSENSDATA(:,6);
    Gyroscope_Z = XSENSDATA(:,7);

    raw_quaternion = XSENSDATA(:,8:11);  
    Ref_quaternion = quaternConj(raw_quaternion);    %conjugate
  
    raw_g_acc = quaternRotate(Accelerometer/9.8, quaternConj(Ref_quaternion));    
    raw_linAcc_2 = raw_g_acc - [zeros(row, 2) ones(row, 1)];     

    raw_linAcc_3 = (raw_g_acc - [zeros(row, 2) ones(row, 1)])*9.8;   
    raw_linAcc_2_z= raw_linAcc_2(:,3)*9.8;     

    %Normal Speed:
    filtCutOff =1.8;
    [b, a] = butter(12, (2*filtCutOff)/(1/SamplePeriod), 'low');   
    acc_Filt = filtfilt(b, a,raw_linAcc_2_z );  
    acc_Filt_total = filtfilt(b, a,raw_linAcc_3 );   

    % LR
    gyr_y = Gyroscope_Y*180/pi;  %to бу
    %Thigh
    % gyr_z = Gyroscope_Z*180/pi;  

    %Normal Speed:
    filtCutOff =1.8;
    [b, a] = butter(12, (2*filtCutOff)/(1/SamplePeriod), 'low');
    gyr_magFilt = filtfilt(b, a, gyr_y);
    gyr_magFilt_total = filtfilt(b,a ,Gyroscope_angle );

    axis(1) = subplot(3,1,1);
    hold on;
    plot(XSENS.time,gyr_y, 'r',XSENS.time,gyr_magFilt, 'g');
    legend('Raw','Filter');
    xlabel('Time (s)');
    ylabel('Angular velocity (deg/s)');
    title('Gyroscope');
    hold off;
%---------------------------------------------------------------
    x = strsplit(XSENSPath (i).name,'.');
    x1 =cell2mat(x(1,1));
    x2 = strcat(Fix_Path,x1 ,'.xlsx');

    all_1 = [raw_linAcc_2_z';gyr_y';acc_Filt';gyr_magFilt'];
    all_2 = all_1';
    xlswrite(x2 , all_2);   
%------------------------------------------------------------------
end













