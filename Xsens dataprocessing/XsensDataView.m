% AHRS_main.m   
%
% This script demonstrates basic usage of the x-IMU MATLAB Library showing
% how data can be easily imported visualised and the library class
% structures are used to organise and access data.

%% Start of script 
addpath('ximu_matlab_library');     % include library
% addpath('quaternion_library');      % include quaternion library
%addpath('felix_library');      % include user definition library
close all;                          % close all figures
clear;                              % clear all variables
clc;                                % clear the command terminal

% Import data and set constant.
% run XSENS_Adapter  MT_2016-11-25_009.txt
if 1
    XSENSPath = 'MT_2023-04-03-012.txt';  % MT_2016-11-24_000_sample.txt,,,MT_2017-06-06_006_cut.txt  MT_2016-11-25_009_cut.txt MT_2017-06-08_000.txt
    onlineflag = 1; 
    run XSENS_Adapter
    %save('Xsens_data','XSENS'); % 'YSKIMU',
%
    %load Xsens_data.mat
end

time = XSENS.time;
Accelerometer = XSENS.Accelerometer;
Gyroscope = XSENS.Gyroscope;
% Magnetometer = XSENS.Magnetometer;
% datalength = size(XSENS.Gyroscope,1);
% Process sensor data through algorithm


%% End of script