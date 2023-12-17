clear;
close all;
clc;
addpath('quaternion_library');  
addpath('ximu_matlab_library');
%import data   
% XSENSPath = 'cx-LR-ns-imu-010.txt'; 
XSENS_path = 'C:\Users\Administrator\Desktop\cx\XSENS data viewV1.0 20170822 (1)\XSENS data viewV1.0 20170822\�ζ�\';
XSENS_name = '*.txt';
XSENSPath = dir([XSENS_path, XSENS_name]);
Fix_Path = 'C:\Users\Administrator\Desktop\cx\XSENS data viewV1.0 20170822 (1)\XSENS data viewV1.0 20170822\�ζ�-4����\';

Length = length(XSENSPath );    %�����ļ�����xls�ĵ��ĸ���
for i = 1:Length        %������ȡ�ļ������ݲ�����
    SamplePeriod = 1/100;
    temp = importdata(strcat(XSENS_path ,XSENSPath (i).name), '\t', 6);  %tab,��7�п�ʼΪ��ֵ
    XSENSDATA = temp.data;
    raw_data = XSENSDATA(:,2:11);
    row=size(XSENSDATA,1);%��һ��ά��
    XSENS = struct;
    XSENS.time = ((0:row-1)*SamplePeriod)';
    %ԭʼ3����ٶȡ����ٶ�
    Accelerometer = XSENSDATA(:,2:4);       % xsens��λ��m/s2
    Accelerometer_X= XSENSDATA(:,2);   
    Accelerometer_Y = XSENSDATA(:,3); 
    Accelerometer_Z = XSENSDATA(:,4); 

    Gyroscope = XSENSDATA(:,5:7);       % ����ǻ����ƽ��ٶ�   
    Gyroscope_angle = XSENSDATA(:,5:7)*180/pi;  %ԭʼ3�������ǽ��ٶ�
    Gyroscope_X = XSENSDATA(:,5);
    Gyroscope_Y = XSENSDATA(:,6);
    Gyroscope_Z = XSENSDATA(:,7);

    raw_quaternion = XSENSDATA(:,8:11);  
    Ref_quaternion = quaternConj(raw_quaternion);    %ȡ�乲�У��Xsens����������Ԫ��
    %ȥ����������������ϵ�´�ֱ�����ϵļ��ٶ�
    raw_g_acc = quaternRotate(Accelerometer/9.8, quaternConj(Ref_quaternion));     % �Ѽ��ٶ�ͨ����Ԫ����ת����������ϵ��
    raw_linAcc_2 = raw_g_acc - [zeros(row, 2) ones(row, 1)];     %����1158�У�3�е����ݣ���һ����0,0,1

    raw_linAcc_3 = (raw_g_acc - [zeros(row, 2) ones(row, 1)])*9.8;    %ԭʼ3����ٶ�
    raw_linAcc_2_z= raw_linAcc_2(:,3)*9.8;     %ȥ������ԭʼ��ֱ�����ϵļ��ٶ�

    %Normal Speed:
    filtCutOff =1.8;
    [b, a] = butter(12, (2*filtCutOff)/(1/SamplePeriod), 'low');   
    acc_Filt = filtfilt(b, a,raw_linAcc_2_z );   %������˹�˲�����ٶ�
    acc_Filt_total = filtfilt(b, a,raw_linAcc_3 );   %3����ٶ��˲�


    % %Slow Speed��
    % filtCutOff =0.8;
    % [b, a] = butter(9, (2*filtCutOff)/(1/SamplePeriod), 'low');   
    % acc_Filt = filtfilt(b, a,raw_linAcc_2_z );   %������˹�˲�����ٶ�

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
    gyr_y = Gyroscope_Y*180/pi;  %����ת��Ϊ�Ƕ�
    %������
    % gyr_z = Gyroscope_Z*180/pi;  %����ת��Ϊ�Ƕ�

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













