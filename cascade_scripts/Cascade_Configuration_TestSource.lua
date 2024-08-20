
--[[
Sequence being followed

A. CONFIGURATION
1. Connecting to TDA
1. Configuring Master from SOP till Channel Config
2. Configuring Slave (i) sequentially from SOP till SPI Connect. i = 1, 2, 3
3. Configuring Slaves together from F/W download till Channel Config
4. Configuring all devices together from LDO Bypass till Frame Config

NOTE:
Update the following in the script accordingly before running
1. metaImage F/W path on line 33
2. TDA Host Board IP Address on line 38
--]]
    
----------------------------------------User Constants----------------------------------------------------
       
dev_list						=	{1, 2, 4, 8}                    -- Device map
RadarDevice						=	{1, 1, 0, 0}				    -- {dev1, dev2, dev3, dev4}, 1: Enable, 0: Disable
	
cascade_mode_list				=	{1, 2, 2, 2}				    -- 0: Single chip, 1: Master, 2: Slave

-- F/W Download Path

-- Uncomment the next line if you wish to pop-up a dialog box to select the firmware image file
-- Otherwise, hardcode the path to the firmware metaimage below
-- By default, the firmware filename is: xwr12xx_metaImage.bin
--
-- metaImagePath                   =   RSTD.BrowseForFile(RSTD.GetSettingsPath(), "bin", "Browse to .bin file")

metaImagePath                   =   "C:\\ti\\mmwave_dfp_01_02_06_03\\firmware\\xwr12xx_metaImage.bin"

-- IP Address for the TDA2 Host Board
-- Change this accordingly for your setup

TDA_IPAddress                   =   "192.168.33.180"

-- Device map of all the devices to be enabled by TDA
-- 1 - master ; 2- slave1 ; 4 - slave2 ; 8 - slave3

deviceMapOverall                =   RadarDevice[1] + (RadarDevice[2]*2) + (RadarDevice[3]*4) + (RadarDevice[4]*8)
deviceMapSlaves                 =   (RadarDevice[2]*2) + (RadarDevice[3]*4) + (RadarDevice[4]*8)

-- Enable/Disable Test Source
-- This is useful during bringup

test_source_enable              =   1                               -- 0: Disable, 1: Enable    
    
------------------------------------------- Sensor Configuration ------------------------------------------------

-- The sensor configuration consists of 3 sections:
-- 		1) Profile Configuration (common to all 4 AWR devices)
--		2) Chirp Configuration (common to all 4 AWR devices)
--		3) Frame Configuration (common to all 4 AWR devices, except for the trigger mode for the master)
-- Change the values below as needed.

-- Profile configuration
local profile_indx              =   0
local start_freq				=	77								-- GHz
local slope						=	79  							-- MHz/us
local idle_time					=	5								-- us
local adc_start_time			=	6								-- us
local adc_samples				=	256							    -- Number of samples per chirp
local sample_freq				=	8000							-- ksps
local ramp_end_time				=	40								-- us
local rx_gain					=	48								-- dB
local tx0OutPowerBackoffCode    =   0
local tx1OutPowerBackoffCode    =   0
local tx2OutPowerBackoffCode    =   0
local tx0PhaseShifter           =   0
local tx1PhaseShifter           =   0
local tx2PhaseShifter           =   0
local txStartTimeUSec           =   0
local hpfCornerFreq1            =   0                               -- 0: 175KHz, 1: 235KHz, 2: 350KHz, 3: 700KHz
local hpfCornerFreq2            =   0                               -- 0: 350KHz, 1: 700KHz, 2: 1.4MHz, 3: 2.8MHz

-- Frame configuration	
local start_chirp_tx			=	0
local end_chirp_tx				=	11
local nchirp_loops				=	128								-- Number of chirps per frame
local nframes_master			=	10							    -- Number of Frames for Master
local nframes_slave			    =	10							    -- Number of Frames for Slaves
local Inter_Frame_Interval		=	100								-- ms
local trigger_delay             =   0                               -- us
local nDummy_chirp              =   0       
local trig_list					=	{1,2,2,2}	                    -- 1: Software trigger, 2: Hardware trigger    
    
     
------------------------------------------- API Configuration ------------------------------------------------    
    
-- 1. Connection to TDA. 2. Selecting Cascade/Single Chip.  3. Selecting 2-chip/4-chip

WriteToLog("Setting up Studio for Cascade started..\n", "blue")

if(0 == ar1.ConnectTDA(TDA_IPAddress, 5001, deviceMapOverall)) then
    WriteToLog("ConnectTDA Successful\n", "green")
else
    WriteToLog("ConnectTDA Failed\n", "red")
    return -1
end

if(0 == ar1.selectCascadeMode(1)) then
    WriteToLog("selectCascadeMode Successful\n", "green")
else
    WriteToLog("selectCascadeMode Failed\n", "red")
    return -1
end

WriteToLog("Setting up Studio for Cascade ended..\n", "blue")
         
--Master Initialization
      
-- SOP Mode Configuration
if (0 == ar1.SOPControl_mult(1, 4)) then
	WriteToLog("Master : SOP Reset Successful\n", "green")
else
	WriteToLog("Master : SOP Reset Failed\n", "red")
	return -1
end
		
-- SPI Connect		
if (0 == ar1.PowerOn_mult(1, 0, 1000, 0, 0)) then
	WriteToLog("Master : SPI Connection Successful\n", "green")
else
	WriteToLog("Master : SPI Connection Failed\n", "red")
	return -1
end

-- Firmware Download. (SOP 4 - MetaImage)
if (0 == ar1.DownloadBssFwOvSPI_mult(1, metaImagePath)) then
	WriteToLog("Master : FW Download Successful\n", "green")
else
	WriteToLog("Master : FW Download Failed\n", "red")
	return -1
end

         
-- RF Power Up
if (0 == ar1.RfEnable_mult(1)) then
	WriteToLog("Master : RF Power Up Successful\n", "green")
else
	WriteToLog("Master : RF Power Up Failed\n", "red")
	return -1
end			
         
-- Channel & ADC Configuration
if (0 == ar1.ChanNAdcConfig_mult(1,1,1,1,1,1,1,1,2,1,0,1)) then
	WriteToLog("Master : Channel & ADC Configuration Successful\n", "green")
else
	WriteToLog("Master : Channel & ADC Configuration Failed\n", "red")
	return -2
end
    
-- Slaves Initialization
   
for i=2,table.getn(RadarDevice) do 
	local status	=	0		        
	if ((RadarDevice[1]==1) and (RadarDevice[i]==1)) then
      
        -- SOP Mode Configuration
		if (0 == ar1.SOPControl_mult(dev_list[i], 4)) then
			WriteToLog("Device "..i.." : SOP Reset Successful\n", "green")
		else
			WriteToLog("Device "..i.." : SOP Reset Failed\n", "red")
			return -1
		end
				
		-- SPI Connect	
		if (0 == ar1.AddDevice(dev_list[i])) then
			WriteToLog("Device "..i.." : SPI Connection Successful\n", "green")
		else
			WriteToLog("Device "..i.." : SPI Connection Failed\n", "red")
			return -1
		end
           
	end
end  
    
-- Firmware Download. (SOP 4 - MetaImage)
if (0 == ar1.DownloadBssFwOvSPI_mult(deviceMapSlaves, metaImagePath)) then
	WriteToLog("Slaves : FW Download Successful\n", "green")
else
	WriteToLog("Slaves : FW Download Failed\n", "red")
	return -1
end
         
-- RF Power Up
if (0 == ar1.RfEnable_mult(deviceMapSlaves)) then
	WriteToLog("Slaves : RF Power Up Successful\n", "green")
else
	WriteToLog("Slaves : RF Power Up Failed\n", "red")
	return -1
end	
         
-- Channel & ADC Configuration
if (0 == ar1.ChanNAdcConfig_mult(deviceMapSlaves,1,1,1,1,1,1,1,2,1,0,2)) then
	WriteToLog("Slaves : Channel & ADC Configuration Successful\n", "green")
else
	WriteToLog("Slaves : Channel & ADC Configuration Failed\n", "red")
	return -2
end
			
-- All devices together        
          
-- Including this depends on the type of board being used.
-- LDO configuration
if (0 == ar1.RfLdoBypassConfig_mult(deviceMapOverall, 3)) then
	WriteToLog("LDO Bypass Successful\n", "green")
else
	WriteToLog("LDO Bypass failed\n", "red")
	return -2
end

-- Low Power Mode Configuration
if (0 == ar1.LPModConfig_mult(deviceMapOverall,0, 0)) then
	WriteToLog("Low Power Mode Configuration Successful\n", "green")
else
	WriteToLog("Low Power Mode Configuration failed\n", "red")
	return -2
end

-- Miscellaneous Control Configuration
if (0 == ar1.SetMiscConfig_mult(deviceMapOverall, 1)) then
	WriteToLog("Misc Control Configuration Successful\n", "green")
else
	WriteToLog("Misc Control Configuration failed\n", "red")
	return -2
end

-- Edit this API to enable/disable the boot time calibration. Enabled by default.
-- RF Init Calibration Configuration
if (0 == ar1.RfInitCalibConfig_mult(deviceMapOverall, 1, 1, 1, 1, 1, 1, 1, 65537)) then
	WriteToLog("RF Init Calibration Successful\n", "green")
else
	WriteToLog("RF Init Calibration failed\n", "red")
	return -2
end
         
-- RF Init
if (0 == ar1.RfInit_mult(deviceMapOverall)) then
	WriteToLog("RF Init Successful\n", "green")
else
	WriteToLog("RF Init failed\n", "red")
	return -2
end

---------------------------Data Configuration----------------------------------
		
-- Data path Configuration
if (0 == ar1.DataPathConfig_mult(deviceMapOverall, 0, 1, 0)) then
	WriteToLog("Data Path Configuration Successful\n", "green")
else
	WriteToLog("Data Path Configuration failed\n", "red")
	return -3
end

-- Clock Configuration
if (0 == ar1.LvdsClkConfig_mult(deviceMapOverall, 1, 1)) then
	WriteToLog("Clock Configuration Successful\n", "green")
else
	WriteToLog("Clock Configuration failed\n", "red")
	return -3
end

-- CSI2 Configuration
if (0 == ar1.CSI2LaneConfig_mult(deviceMapOverall, 1, 0, 2, 0, 4, 0, 5, 0, 3, 0)) then
	WriteToLog("CSI2 Configuration Successful\n", "green")
else
	WriteToLog("CSI2 Configuration failed\n", "red")
	return -3
end

----------------------------Test Source Configuration------------------------------
-- This is useful for initial bringup.
-- Each device is configured with a test object at a different location.
	
if(test_source_enable == 1) then
    
    if(RadarDevice[1] == 1) then
        -- Object at 5 m with x = 4m and y = 3m
        if (0 == ar1.SetTestSource_mult(1, 3, 4, 0, 0, 0, 0, -0.36, 0, -0.66, 0.36, 0, 0.66, -2.5, 4, 3, 0, 0, 0, 0, -0.32, 0, -0.46, 0.32, 0, 0.46, -2.5, 0, 0, 0.5, 0, 1, 0, 1.5, 0, 0, 0, 0, 0, 0,0)) then
            WriteToLog("Device 1 : Test Source Configuration Successful\n", "green")
        else
            WriteToLog("Device 1 : Test Source Configuration failed\n", "red")
            return -3
        end
    end
    
    if(RadarDevice[2] == 1) then        
        -- Object at 5 m with x = 3m and y = 4m
        if (0 == ar1.SetTestSource_mult(1, 3, 4, 0, 0, 0, 0, -0.36, 0, -0.66, 0.36, 0, 0.66, -2.5, 4, 3, 0, 0, 0, 0, -0.32, 0, -0.46, 0.32, 0, 0.46, -2.5, 0, 0, 0.5, 0, 1, 0, 1.5, 0, 0, 0, 0, 0, 0,0)) then
            WriteToLog("Device 2 : Test Source Configuration Successful\n", "green")
        else
            WriteToLog("Device 2 : Test Source Configuration failed\n", "red")
            return -3
        end
    end
    
    if(RadarDevice[3] == 1) then         
        -- Object at 13 m with x = 12m and y = 5m
        if (0 == ar1.SetTestSource_mult(4, 12, 5, 0, 0, 0, 0, -327, 0, -327, 327, 327, 327, -2.5, 327, 327, 0, 0, 0, 0, -327, 0, -327, 327, 327, 327, -95, 0, 0, 0.5, 0, 1, 0, 1.5, 0, 0, 0, 0, 0, 0,0)) then
            WriteToLog("Device 3 : Test Source Configuration Successful\n", "green")
        else
            WriteToLog("Device 3 : Test Source Configuration failed\n", "red")
            return -3
        end
    end
    
    if(RadarDevice[4] == 1) then        
        -- Object at 13 m with x = 5m and y = 12m
        if (0 == ar1.SetTestSource_mult(8, 5, 12, 0, 0, 0, 0, -327, 0, -327, 327, 327, 327, -2.5, 327, 327, 0, 0, 0, 0, -327, 0, -327, 327, 327, 327, -95, 0, 0, 0.5, 0, 1, 0, 1.5, 0, 0, 0, 0, 0, 0,0)) then
            WriteToLog("Device 4 : Test Source Configuration Successful\n", "green")
        else
            WriteToLog("Device 4 : Test Source Configuration failed\n", "red")
            return -3
        end
    end
       
end           
			
---------------------------Sensor Configuration-------------------------
				
-- Profile Configuration
if (0 == ar1.ProfileConfig_mult(deviceMapOverall, 0, start_freq, idle_time, adc_start_time, ramp_end_time, 0, 0, 0, 0, 0, 0, slope, 0, adc_samples, sample_freq, 0, 0, rx_gain)) then
	WriteToLog("Profile Configuration successful\n", "green")
else
	WriteToLog("Profile Configuration failed\n", "red")
	return -4
end
	
-- Chirp Configuration
-- Chirp 0
if (0 == ar1.ChirpConfig_mult(deviceMapOverall, start_chirp_tx, end_chirp_tx, 0, 0, 0, 0, 0, 1, 1, 0)) then
	WriteToLog("Chirp 0 Configuration successful\n", "green")
else
	WriteToLog("Chirp 0 Configuration failed\n", "red")
	return -4
end
	
-- Enabling/ Disabling Test Source
if(test_source_enable == 1) then
    ar1.EnableTestSource_mult(deviceMapOverall, 1)
    WriteToLog("Enabling Test Source Configuration successful\n", "green")
end
    
-- Frame Configuration               
-- Master
if (0 == ar1.FrameConfig_mult(1,start_chirp_tx,end_chirp_tx,nframes_master, nchirp_loops, Inter_Frame_Interval, 0, 0, 1)) then
    WriteToLog("Master : Frame Configuration successful\n", "green")
else
    WriteToLog("Master : Frame Configuration failed\n", "red")
end
-- Slaves 
if (0 == ar1.FrameConfig_mult(deviceMapSlaves,start_chirp_tx,end_chirp_tx,nframes_slave, nchirp_loops, Inter_Frame_Interval, 0, 0, 2)) then
    WriteToLog("Slaves : Frame Configuration successful\n", "green")
else
    WriteToLog("Slaves : Frame Configuration failed\n", "red")
end