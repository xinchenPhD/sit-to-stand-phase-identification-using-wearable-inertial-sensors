clear;
close all;
clc;
addpath('quaternion_library');  
addpath('ximu_matlab_library');
%import data   
% XSENSPath = 'cx-LR-ns-imu-010.txt'; 
XSENS_path = 'C:\Users\Administrator\Desktop\cx\XSENS data viewV1.0 20170822 (1)\XSENS data viewV1.0 20170822\晃动\';
XSENS_name = '*.txt';
XSENSPath = dir([XSENS_path, XSENS_name]);
Fix_Path = 'C:\Users\Administrator\Desktop\cx\XSENS data viewV1.0 20170822 (1)\XSENS data viewV1.0 20170822\晃动-4特征\';

Length = length(XSENSPath );    %计算文件夹里xls文档的个数
for i = 1:Length        %批量读取文件的内容并保存
    SamplePeriod = 1/100;
    temp = importdata(strcat(XSENS_path ,XSENSPath (i).name), '\t', 6);  %tab,第7行开始为数值
    XSENSDATA = temp.data;
    raw_data = XSENSDATA(:,2:11);
    row=size(XSENSDATA,1);%第一个维度
    XSENS = struct;
    XSENS.time = ((0:row-1)*SamplePeriod)';
    %原始3轴加速度、角速度
    Accelerometer = XSENSDATA(:,2:4);       % xsens单位是m/s2
    Accelerometer_X= XSENSDATA(:,2);   
    Accelerometer_Y = XSENSDATA(:,3); 
    Accelerometer_Z = XSENSDATA(:,4); 

    Gyroscope = XSENSDATA(:,5:7);       % 这个是弧度制角速度   
    Gyroscope_angle = XSENSDATA(:,5:7)*180/pi;  %原始3轴陀螺仪角速度
    Gyroscope_X = XSENSDATA(:,5);
    Gyroscope_Y = XSENSDATA(:,6);
    Gyroscope_Z = XSENSDATA(:,7);

    raw_quaternion = XSENSDATA(:,8:11);  
    Ref_quaternion = quaternConj(raw_quaternion);    %取其共轭，校正Xsens求解出来的四元数
    %去除重力，世界坐标系下垂直方向上的加速度
    raw_g_acc = quaternRotate(Accelerometer/9.8, quaternConj(Ref_quaternion));     % 把加速度通过四元数旋转到世界坐标系下
    raw_linAcc_2 = raw_g_acc - [zeros(row, 2) ones(row, 1)];     %构造1158行，3列的数据，归一化的0,0,1

    raw_linAcc_3 = (raw_g_acc - [zeros(row, 2) ones(row, 1)])*9.8;    %原始3轴加速度
    raw_linAcc_2_z= raw_linAcc_2(:,3)*9.8;     %去重力后原始垂直方向上的加速度

    %Normal Speed:
    filtCutOff =1.8;
    [b, a] = butter(12, (2*filtCutOff)/(1/SamplePeriod), 'low');   
    acc_Filt = filtfilt(b, a,raw_linAcc_2_z );   %巴特沃斯滤波后加速度
    acc_Filt_total = filtfilt(b, a,raw_linAcc_3 );   %3轴加速度滤波


    % %Slow Speed：
    % filtCutOff =0.8;
    % [b, a] = butter(9, (2*filtCutOff)/(1/SamplePeriod), 'low');   
    % acc_Filt = filtfilt(b, a,raw_linAcc_2_z );   %巴特沃斯滤波后加速度

    % sm = smooth(acc_Filt,    'sgolay'  );
    % %            'moving'   - Moving average (default)
    % %           'lowess'   - Lowess (linear fit)
    % %           'loess'    - Loess (quadratic fit)
    % %           'sgolay'   - Savitzky-Golay
    % %           'rlowess'  - Robust Lowess (linear fit)
    %           'rloess'   - Robust Loess (quadratic fit)


%     %plot
%     figure('Name', 'XSENS Sensor Data');
%     axis(2) = subplot(3,1,2);
%     hold on;
%     plot(XSENS.time, raw_linAcc_2_z, 'r',XSENS.time, acc_Filt, 'g');
%     legend('Raw','Filter');
%     xlabel('Time (s)');
%     ylabel('Acceleration (m/s2)');
%     title('Accelerometer');
%     hold off;

    % LR
    gyr_y = Gyroscope_Y*180/pi;  %弧度转化为角度
    %大腿上
    % gyr_z = Gyroscope_Z*180/pi;  %弧度转化为角度

    %Normal Speed:
    filtCutOff =1.8;
    [b, a] = butter(12, (2*filtCutOff)/(1/SamplePeriod), 'low');
    gyr_magFilt = filtfilt(b, a, gyr_y);
    gyr_magFilt_total = filtfilt(b,a ,Gyroscope_angle );

    % %Slow Speeed:
    % filtCutOff = 1.8;
    % [b, a] = butter(10, (2*filtCutOff)/(1/SamplePeriod), 'low');
    % gyr_magFilt = filtfilt(b, a, gyr_y);



    % % HP filter accelerometer data
    % filtCutOff = 0.001;
    % [b, a] = butter(5, (2*filtCutOff)/(1/SamplePeriod), 'high');
    % gyr_magFilt = filtfilt(b, a, gyr);


    % axis(1) = subplot(3,1,1);
    % hold on;
    % plot(XSENS.time,gyr_y, 'r',XSENS.time,gyr_magFilt, 'g');
    % legend('Raw','Filter');
    % xlabel('Time (s)');
    % ylabel('Angular velocity (deg/s)');
    % title('Gyroscope');
    % hold off;
    
    
    
%---------------------------------------------------------------
    x = strsplit(XSENSPath (i).name,'.');
    x1 =cell2mat(x(1,1));
    x2 = strcat(Fix_Path,x1 ,'.xlsx');

    all_1 = [raw_linAcc_2_z';gyr_y';acc_Filt';gyr_magFilt'];
    all_2 = all_1';
    xlswrite(x2 , all_2);   
%----------------------------------------
    
end













