import os
import sys

srcpath = os.path.realpath('SourceFiles')
sys.path.append(srcpath)

import warnings # this is for GUI warnings
warnings.filterwarnings("ignore")

from tevisainst import TEVisaInst

import numpy as np

import keyboard
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

#init DAC
#Set rates for DAC 
segLen = 128
sampleRateDAC = 9E9
PulseWidth = segLen / sampleRateDAC

print("Pulse Width: ")
print(PulseWidth)

ncoFreq= 2.4E9

max_dac = (2**16)-1

#Preallocate
dacWaveI = []
dacWaveQ = []

'''
# Connect to instrument(PXI)
sid = 12 #PXI slot of AWT on chassis
from teproteus import TEProteusAdmin as TepAdmin
admin = TepAdmin() #required to control PXI module
inst = admin.open_instrument(slot_id=sid) 
resp = inst.send_scpi_query("*IDN?") # Get the instrument's *IDN
print('connected to: ' + resp) # Print *IDN
'''
# Connect to instrument(LAN)

# inst_addr = 'TCPIP::192.168.1.9::5025::SOCKET' 
inst_addr = 'TCPIP::169.254.3.195::5025::SOCKET'

inst = TEVisaInst(inst_addr)
resp = inst.send_scpi_query("*IDN?")
print('connected to: ' + resp)

# initialize DAC
inst.send_scpi_cmd('*CLS; *RST')   

cmd = ':SOUR:MODE DIR'
resp = inst.send_scpi_cmd(cmd)     

#AWG channel
ch = 1
cmd = ':INST:CHAN {0}'.format(ch)
inst.send_scpi_cmd(cmd)

print('CH I DAC Clk Freq {0}'.format(sampleRateDAC)) 
cmd = ':FREQ:RAST {0}'.format(sampleRateDAC)
inst.send_scpi_cmd(cmd)
inst.send_scpi_cmd(':INIT:CONT ON')
inst.send_scpi_cmd(':TRAC:DEL:ALL')
   
def makeDCData(segLen):
    global dacWaveI
    global dacWaveQ
    
    print(max_dac)

    half_dac=max_dac/2
    data_type = np.uint16
    
    #Set DC
    dacWaveDC = np.zeros(segLen) + half_dac    
    dacWaveI = dacWaveDC.astype(data_type)   
    
    
def makeDC_On_Data(segLen):
    global dacWaveI
    global dacWaveQ
    
    half_dac=max_dac/2
    data_type = np.uint16
    
    print(max_dac)
    
    #Set DC
    dacWaveDC = np.zeros(segLen) + half_dac
    dacWaveDC[0:10] = max_dac * 0.9
    dacWaveI = dacWaveDC.astype(data_type)   

    plt.plot(dacWaveI)  
    plt.show()

def downLoad(segnum):
    global dacWaveI

    dacWaveIQ = dacWaveI
    
    print(len(dacWaveIQ))
    
    inst.send_scpi_cmd('*CLS') 
    
    print('Segment {0} Length {1}'.format(segnum, len(dacWaveIQ)))

    # Define segment
    cmd = ':TRAC:DEF {0}, {1}'.format(segnum, len(dacWaveIQ))
    inst.send_scpi_cmd(cmd)
    
    resp = inst.send_scpi_query(':SYSTem:INFormation:DAC?')
    print("DAC = ")
    print(resp)
    

    # Select the segment
    cmd = ':TRAC:SEL {0}'.format(segnum)
    inst.send_scpi_cmd(cmd)
    
    resp = inst.send_scpi_query(':SYST:ERR?')
    print("Trace Sel Error = ")
    print(resp)

    # Increase the timeout before writing binary-data:
    inst.timeout = 30000
    # Send the binary-data with *OPC? added to the beginning of its prefix.
    inst.write_binary_data(':TRAC:DATA', dacWaveIQ)
    # Set normal timeout
    inst.timeout = 10000

    resp = inst.send_scpi_query(':SYST:ERR?')
    print("Trace Download Error = ")
    print(resp)
    


def setTask():
   
   
    cmd = ':TASK:COMP:LENG 2'
    inst.send_scpi_cmd(cmd)
     
    cmd = ':TASK:COMP:SEL 1' 
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:SEGM 1'
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:LOOP 1'
    inst.send_scpi_cmd(cmd)
    # cmd = ':TASK:COMP:DTR ON'
    # inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:NEXT1 2'
    inst.send_scpi_cmd(cmd)
    
    cmd = ':TASK:COMP:SEL 2' 
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:SEGM 2'
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:LOOP 1000'
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:NEXT1 1'
    inst.send_scpi_cmd(cmd)

    cmd = ':TASK:COMP:WRITE'
    inst.send_scpi_cmd(cmd)
    cmd = ':SOUR:FUNC:MODE TASK'
    inst.send_scpi_cmd(cmd)
    

ch=1
# Select channel
cmd = ':INST:CHAN {0}'.format(ch)
inst.send_scpi_cmd(cmd)

makeDC_On_Data(segLen)
downLoad(1)
makeDCData(segLen)
downLoad(2)
#set_duc(ncoFreq)
#downLoad_mrkr()
setTask()

cmd = ':OUTP ON'
rc = inst.send_scpi_cmd(cmd)


# # Init ADC anf GUI



# breakVal = 0
# dcOff = 0
# spectrumInv = 0

# #Set rates for DAC and ADC
# sampleRateADC = 5.4E9

# #Set number of frames to be collected
# numframes, framelen = 1, 9600*10
# totlen = numframes * framelen

# #Preallocate processing memory
# wav1 = np.zeros(framelen, dtype=np.uint16)
# xT = np.linspace(0, numframes * framelen,  numframes * framelen )
# xT = xT/sampleRateADC
# tpCount = len(wav1)
# timeStep = xT[1]-xT[0]
# xF = np.fft.fftfreq(tpCount, timeStep)
# xF = xF[range(int(len(wav1)/2))]

# # Start the digitizer

#   # initialize DAC and take the first capture.
# inst.send_scpi_cmd(':DIG:MODE SING')

# print('ADC Clk Freq {0}'.format(sampleRateADC))
# cmd = ':DIG:FREQ  {0}'.format(sampleRateADC)
# inst.send_scpi_cmd(cmd)

# # Enable capturing data from channel 1
# inst.send_scpi_cmd(':DIG:CHAN:SEL 1')
# inst.send_scpi_cmd(':DIG:CHAN:STATE ENAB')
# # Select the internal-trigger as start-capturing trigger:
# inst.send_scpi_cmd(':DIG:TRIG:SOURCE CPU')

# cmd = ':DIG:ACQuire:FRAM:DEF {0},{1}'.format(numframes, framelen)
# inst.send_scpi_cmd(cmd)

# # Select the frames for the capturing 
# # (all the four frames in this example)
# capture_first, capture_count = 1, numframes
# cmd = ':DIG:ACQuire:FRAM:CAPT {0},{1}'.format(capture_first, capture_count)
# inst.send_scpi_cmd(cmd)

# # Start the digitizer's capturing machine
# inst.send_scpi_cmd(':DIG:INIT ON')
# inst.send_scpi_cmd(':DIG:TRIG:IMM')
# inst.send_scpi_cmd(':DIG:INIT OFF')

# # Choose which frames to read (all in this example)
# inst.send_scpi_cmd(':DIG:DATA:SEL ALL')

# # Choose what to read 
# # (only the frame-data without the header in this example)
# inst.send_scpi_cmd(':DIG:DATA:TYPE FRAM')

# # Get the total data size (in bytes)
# resp = inst.send_scpi_query(':DIG:DATA:SIZE?')
# num_bytes = np.uint32(resp)
# print('Total read size in bytes: ' + resp)
# print()

# # Read the data that was captured by channel 1:
# inst.send_scpi_cmd(':DIG:CHAN:SEL 1')
# wavlen = num_bytes // 2
# rc = inst.read_binary_data(':DIG:DATA:READ?', wav1, num_bytes)

# fourierTransform = np.fft.fft(wav1-dcOff)/len(wav1)           # Normalize amplitude
# fourierTransform = abs(fourierTransform[range(int(len(wav1)/2))]) # Exclude sampling frequency

# if(spectrumInv == 1):
#     fftPlot = np.log10(fourierTransform[::-1])
# else:
#     fftPlot = np.log10(fourierTransform)
    

# # GUI - Define the GUI

# # Run GUI event loop
# plt.ion()
  
# # Create sub plots
# figure, ax1 = plt.subplots(2)
# line1, = ax1[0].plot(xT, wav1, color="yellow")
# #line2, = ax1[1].plot(xF[300000:384000], fftPlot[300000:384000], color="yellow")
# line2, = ax1[1].plot(xF[40000:48000], fftPlot[40000:48000], color="yellow")
# #line2, = ax1[1].plot(xF, fftPlot, color="yellow")

# # setting x-axis label and y-axis label
# ax1[0].set(xlabel='Time = (Pts/'+ str(sampleRateDAC)+')', ylabel='Amplitude = (ADCRng/4096)')
# ax1[0].set_position([0.2, 0.55, 0.7, 0.35]) #x, y, w, h]
# ax1[0].set_ylim([0,4096])
# ax1[0].set(facecolor = "black")
# ax1[0].grid()

# ax1[1].set(xlabel='Frequency', ylabel='FFT Amplitude')
# ax1[1].set_position([0.2, 0.1, 0.7, 0.35]) #x, y, w, h]
# ax1[1].set_ylim([-4,1])
# #ax1[1].set_xlim([0,60e6])
# ax1[1].set(facecolor = "black")
# ax1[1].grid()

# # GUI - Button actions
  
# def vMax(val):
#     global inst
#     cmd = ':DIG:CHAN:RANG HIGH'
#     inst.send_scpi_cmd(cmd)
#     range = inst.send_scpi_query(':DIG:CHAN:RANG?')
#     print('Range ' + range)
    
# def vMed(val):
#     cmd = ':DIG:CHAN:RANG MED'
#     inst.send_scpi_cmd(cmd)
#     range = inst.send_scpi_query(':DIG:CHAN:RANG?')
#     print('Range ' + range)

# def vMin(val):
#     cmd = ':DIG:CHAN:RANG LOW'
#     inst.send_scpi_cmd(cmd)
#     range = inst.send_scpi_query(':DIG:CHAN:RANG?')
#     print('Range ' + range)
    
# def freeRun(val):
#     cmd = ':DIG:TRIG:SOURCE CPU'
#     inst.send_scpi_cmd(cmd)
#     range = inst.send_scpi_query(':DIG:TRIG:SOURCE?')
#     print('Trigger ' + range)
    
# def trigExt(val):
#     cmd = ':DIG:TRIG:SOURCE TASK1'
#     inst.send_scpi_cmd(cmd)
#     range = inst.send_scpi_query(':DIG:TRIG:SOURCE?')
#     print('Trigger ' + range)    
 
# def dc(val):
#     ax2.set_xticklabels(['', '0Hz', '562MHz', '1124MHz', '1686MHz', '2248MHz', '2810MHz'])
#     global spectrumInv
#     spectrumInv = 0; 

# def two(val):
#     ax2.set_xticklabels(['', '2700MHz', '3268MHz', '3824MHz', '4386MHz', '4948MHz', '5510MHz'])
#     global spectrumInv
#     spectrumInv = 1;

# def five(val):
#     ax2.set_xticklabels(['', '5400MHz', '5962MHz', '6524MHz', '7086MHz', '7648MHz', '8210MHz'])
#     global spectrumInv
#     spectrumInv = 0;

# def eight(val):
#     ax2.set_xticklabels(['', '8100MHz', '8662MHz', '9225MHz', '9787MHz', '10350MHz', '10910MHz'])
#     global spectrumInv
#     spectrumInv = 1;
    
# def exitLoop(val):
#     global breakVal
#     breakVal = 1
    

# xAnchor = 0.04
# yAnchor = 0.33

# ax1_button_500 = plt.axes([xAnchor, 0.85 , 0.03,0.05]) #xposition, yposition, width and height
# grid_button_500 = Button(ax1_button_500, 'Max', color = 'white', hovercolor = 'grey')
# grid_button_500.on_clicked(vMax)

# ax1_button_800 = plt.axes([xAnchor+0.035, 0.85 , 0.03,0.05]) #xposition, yposition, width and height
# grid_button_800 = Button(ax1_button_800, 'Med', color = 'white', hovercolor = 'grey')
# grid_button_800.on_clicked(vMed)

# ax1_button_1000 = plt.axes([xAnchor+(0.035*2), 0.85 , 0.03,0.05]) #xposition, yposition, width and height
# grid_button_1000 = Button(ax1_button_1000, 'Min', color = 'white', hovercolor = 'grey')
# grid_button_1000.on_clicked(vMin)


# yAnchor = 0.7
# ax3_button_free = plt.axes([0.04, yAnchor , 0.1,0.05]) #xposition, yposition, width and height
# grid_button_free = Button(ax3_button_free, 'Free Run', color = 'white', hovercolor = 'grey')
# grid_button_free.on_clicked(freeRun)

# ax3_button_trig = plt.axes([0.04, yAnchor-0.075 , 0.1,0.05]) #xposition, yposition, width and height
# grid_button_trig = Button(ax3_button_trig, 'Trigger', color = 'white', hovercolor = 'grey')
# grid_button_trig.on_clicked(trigExt)

# ax3_button_exit = plt.axes([0.04, yAnchor-(0.075*3) , 0.1,0.05]) #xposition, yposition, width and height
# grid_button_exit = Button(ax3_button_exit, 'Exit', color = 'white', hovercolor = 'grey')
# grid_button_exit.on_clicked(exitLoop)
# # GUI - Define the Buttons

# xAnchor = 0.04
# yAnchor = 0.33

# ax1_button_500 = plt.axes([xAnchor, 0.85 , 0.03,0.05]) #xposition, yposition, width and height
# grid_button_500 = Button(ax1_button_500, 'Max', color = 'white', hovercolor = 'grey')
# grid_button_500.on_clicked(vMax)

# ax1_button_800 = plt.axes([xAnchor+0.035, 0.85 , 0.03,0.05]) #xposition, yposition, width and height
# grid_button_800 = Button(ax1_button_800, 'Med', color = 'white', hovercolor = 'grey')
# grid_button_800.on_clicked(vMed)

# ax1_button_1000 = plt.axes([xAnchor+(0.035*2), 0.85 , 0.03,0.05]) #xposition, yposition, width and height
# grid_button_1000 = Button(ax1_button_1000, 'Min', color = 'white', hovercolor = 'grey')
# grid_button_1000.on_clicked(vMin)


# yAnchor = 0.7
# ax3_button_free = plt.axes([0.04, yAnchor , 0.1,0.05]) #xposition, yposition, width and height
# grid_button_free = Button(ax3_button_free, 'Free Run', color = 'white', hovercolor = 'grey')
# grid_button_free.on_clicked(freeRun)

# ax3_button_trig = plt.axes([0.04, yAnchor-0.075 , 0.1,0.05]) #xposition, yposition, width and height
# grid_button_trig = Button(ax3_button_trig, 'Trigger', color = 'white', hovercolor = 'grey')
# grid_button_trig.on_clicked(trigExt)

# ax3_button_exit = plt.axes([0.04, yAnchor-(0.075*3) , 0.1,0.05]) #xposition, yposition, width and height
# grid_button_exit = Button(ax3_button_exit, 'Exit', color = 'white', hovercolor = 'grey')
# grid_button_exit.on_clicked(exitLoop)



# def acquireData():
    
    
#     #std_dev = 100e6
#     #ncoFreqJump = np.random.normal(loc=ncoFreq, scale=std_dev)
#     #cmd = ':SOUR:NCO:CFR1 {0}'.format(ncoFreqJump)
#     #resp = inst.send_scpi_cmd(cmd)
    
#     wav1 = np.zeros(framelen, dtype=np.uint16)
    
#     # Start the digitizer's capturing machine
#     inst.send_scpi_cmd(':DIG:INIT ON')
#     inst.send_scpi_cmd(':DIG:TRIG:IMM')
#     # Stop the digitizer's capturing machine (to be on the safe side)
#     inst.send_scpi_cmd(':DIG:INIT OFF')

#     # Read the data that was captured by channel 1:
#     inst.send_scpi_cmd(':DIG:CHAN:SEL 1')

#     resp = inst.read_binary_data(':DIG:DATA:READ?', wav1, num_bytes)
#     resp = resp
#     #print("Aquisition Error = ")
#     #print(resp)
    
    
#     wav1 = wav1 - dcOff
#     wav2 = wav1 - np.average(wav1)
#     w = np.blackman(len(wav1))
#     wavFFT = w * wav2    #wavFFT = wav1
#     fourierTransform = np.fft.fft(wavFFT)/len(wav2)           # Normalize amplitude
#     fourierTransform = abs(fourierTransform[range(int(len(wav2)/2))]) # Exclude sampling frequency

#     if(spectrumInv == 1):
#         fftPlot = np.log10(fourierTransform[::-1])
#     else:
#         fftPlot = np.log10(fourierTransform)

#     # Plot the samples
#     # updating data values
#     line1.set_xdata(xT)
#     line1.set_ydata(wav1) #Subtracting offset twice? 
#     #line2.set_xdata(xF[300000:384000])
#     #line2.set_ydata(fftPlot[300000:384000])
    
#     line2.set_xdata(xF[40000:48000])
#     line2.set_ydata(fftPlot[40000:48000])

    
#     # drawing updated values
#     figure.canvas.draw()
  
#     # This will run the GUI event
#     # loop until all UI events
#     # currently waiting have been processed
#     figure.canvas.flush_events()
     
#     time.sleep(0.1)
#     del wav1
#     del fftPlot

# while True:
#     try:
#         if keyboard.is_pressed(' '):
#             print("Stop initiated...")
#             plt.close()
#             break
#         if(breakVal==1):
#             print("Stop initiated...")
#             plt.close()
#             break
    
#         acquireData()    
        
#     except:
#         break


inst.close_instrument()
