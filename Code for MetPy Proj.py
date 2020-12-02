#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary modules
import numpy as np
import pandas as pd
import metpy as mp
import metpy.calc as mpcalc
import metpy.constants as mpconsts
import matplotlib.pyplot as plt
from metpy.units import units
from mpl_toolkits.axisartist import Subplot
from matplotlib.ticker import FuncFormatter, Formatter
from datetime import datetime#, timezone
from siphon.simplewebservice.wyoming import WyomingUpperAir
import time
import statistics

##############
## USER INPUTS
##############
# change date and site to test any archived sounding 
date = '2020081012' #str 'YYYYMMDDHH' of sounding
site =  'ILX' #3 letter station ID (ie ILX or OUN)

## #  # # # # # # # # # # # #  # #
# Data Prep
###########
def readSounding (datestr, station):
    ''' Download the data from the wyoming site and prep for future use
    '''
    #read in the data 
    date= datetime.strptime(datestr, '%Y%m%d%H')
    df=WyomingUpperAir.request_data(date, station)
    
    #drop unnesesary columns from our dataframe 
    df=df.drop(columns=['elevation', 'direction','speed', 'u_wind','v_wind', 'station', 
                        'station_number', 'time', 'latitude','longitude'])
    #shorten the names of the remaining columns
    df=df.rename(columns={"temperature":"tempC", "dewpoint":"dewC", "height":"hght", "pressure":"press"})

    #convert temp and dew to kelvin
    df['tempK'], df['dewK'] = df.tempC+273.15 , df.dewC+273.15
    return df

## #  # # # # # # # # # # # #  # #
##############
# Calculations
##############
def moist_lapse(t_start, press_start, delta_z, Eqn):
    '''Calculating the moist adiabatic lapse rate
    ---
    Inputs: 
        t_start: starting temperature in Kelvin
        press_start: starting pressure in hPa
        delta_z: height in meters
        Eqn: Equation used in the calculation of sat vapor pressure
    
    Outputs: 
        t_final: Final temperature at the next level in Kelvin
       '''
    #Det sat mixing ratio
    Ws= sat_mixing_ratio(press_start, t_start, Eqn)

    #Det moist lapse rate #t_start in kelvin
    g, Lv, Cp_d, Rd, E = mpconsts.g.m, mpconsts.Lv.m, mpconsts.Cp_d.m, (mpconsts.Rd.m*1000),  mpconsts.epsilon.m 
    M_L = g * ( (1 + ((Lv*Ws)/(Rd*t_start))) / (Cp_d + ((np.square(Lv)*Ws*E)/(np.square(t_start)*Rd))) )

    #Det next parcel point
    delta_t = M_L * delta_z #change in t (in K ) across the distance delta_z
    t_final = t_start + delta_t
    return t_final 

#**************
def sat_mixing_ratio(P, T, Eqn):
    """Calc mixing ratio: can specify one of three different sat vap pressure calculation methods
    ---
    Inputs:
        P: Pressure in hPa
        T: Temperature in Kelvin
        Eqn: Equation to use in the calculation of sat vap press
        
    Outputs: 
        Ws: saturation mixing ratio in kg/kg
    """
    #Calc es 
    e_s = sat_vap_press(T, Eqn)
    if Eqn == 'metpy': e_s = e_s.m
    #Calc the mixing ratio using the specified e_s as an input 
    Ws = .622 * (e_s / (P -e_s)) #mixing ratio (kg/kg) 
    return Ws

#**************
def Virtual_Temp( P, T, Eqn):
    """Calc the virtual temp using the saturation mixing ratio that was calculated
    ---
    Inputs: 
        P: Pressure in hPa
        T: Temperature in Kelvin
        Eqn: Equation to use in the calculation of sat vap press
        
    Outputs: 
        TV: Virtual Temperature in Kelvin  
    """
    #Calc the mixing ratio using the specified e_s as an input 
    Ws = sat_mixing_ratio(P, T, Eqn)

    #Convert temperature to virtual temperature (Input and output T is in K)
    TV = T * ( (1 + (Ws/mpconsts.epsilon.m)) / (1 + Ws))
    return TV

#**************
def thetae(P, T, Eqn):
    """Calculate theta-e using Eqn 2 from Bolton (1980)  
    theta_e = (Tc*(pv/pd)**(k_dry)) * np.exp((Lv*w)/(4190*Tc))
    ---
    Inputs: 
        Tc: Temperature in Celcius
        T: Temperature in Kelvin
        P: Pressure in hPa
        Eqn: Equation to use in the calculation of sat vap press
    
    Outputs:
        theta_e: Theta-e in Kelvin
    """
    #Define the constants 
    #Note: We found that Rd was printing out as Rd = 0.287, so we multiplied it by 1000
    Lv, Cp_d, Rd = mpconsts.Lv.m, mpconsts.Cp_d.m, (mpconsts.Rd.m*1000) 
    Cw = 4190 #specific heat of water
    Ws = sat_mixing_ratio(P, T, Eqn)
    Cw_d = Cp_d + (Ws * Cw) 

    #Plug into Eqn 2 from Bolton (1980)
    a = (T * (1000/P)**( Rd/Cw_d ))
    b = np.exp((Lv*Ws) / (Cw_d*T))
    theta_e = a * b
    
    theta_e = a * b
    return theta_e


## #  # # # # # # # # # # # #  # #
#####################
# DCAPE and e_s Calcs
#####################
def sat_vap_press(T, Eqn):
    ''' Calculate the saturation vapor pressure three seperate ways
    ---
    Inputs: 
        T: Temperature in Kelvin 
        Eqn: Equation to use in the saturation vapor pressure calculation
        
    Output:
        e_s: saturation vapor pressure
    '''
    #Used metpy as a way to check our values
    #Note: metpy uses Bolton (1980) in order to calculate saturation vapor pressure
    if Eqn == 'metpy':
        try: T=T.to_numpy()
        except: pass
        
        #Converting units
        Tk= T*units.degK
        e_s = mpcalc.saturation_vapor_pressure(Tk)
        return e_s

    # * * *
    ## Calc e_s using Eqn 23 from Alduchov and Eskridge (1996)
    elif Eqn == 'Magnus':
        Tc= T-273.15 #convert to C
        e_s = 6.1094 *np.exp( (17.625*Tc) / (Tc+243.04) )
        return e_s

    # * * *
    elif Eqn == 'Wexler':
        ## Calc e_s using Eqn from Wexler (1976)
        #Breaking down the equation
        a = ((-2.9912729 *(10**3)) * ((T)**(-2)))
        b = ((6.0170128 *(10**3)) * ((T)**(-1)))
        c = (1.887643845 *(10**1))
        d = ((2.8354721 *(10**-2)) * (T))
        e = ((1.7838301 *(10**-5)) * ((T)**(2)))
        f = ((8.4150417 *(10**-10)) * ((T)**(3)))
        g = ((4.4412543 *(10**-13)) * ((T)**(4)))
        h = ((2.858487 * (np.log(T))))
        #Calculating the saturation vapor pressure
        e_s= 0.01* np.exp( a-b+c-d+e-f+g+h )
        return e_s
     
    # * * *
    elif Eqn == 'Buck':
        ## Calc e_s using Eqn from Buck (1996) (for liquid water only)
        
        # Temperature converted to celcius
        Tc = T - 273.15 
        
        #Break down equation
        a1 = (18.678-(Tc/234.5))
        b1 = (Tc/(257.14+Tc))
        e_s = (6.1121*np.exp(a1*b1))
        return e_s

########
#**************
def DCAPE (data, ES, origin_method):
    """DCAPE will be calcuated using virtual temperature (and therefore the mixing ratio and saturation mixing ratio 
        calculated above. We will be checking to see which version of saturation vapor pressure will result in the most
        accurate DCAPE values
    ---
    Inputs:
        data: the sounding dataset that is read in 
        ES: Noting which saturation vapor pressure equation you would like to use
        origin_method: Noting which calculation you would like to use to find the source origin height 
                       (upper bounda of integration) for DCAPE
    
    Outputs:
        DCAPE: Downdraft CAPE (Convective Available Potential Engery) in J/kg
        calculation_time: The time it takes for the code to run; will help determine if the options are computationally expensive
    """
    
    calc_start_time = time.time()
    def DCAPE_origin(data, Eqn, method):
        """ Will let you choose between 3 different options for finding the source origin height
        ---
        Inputs:
            data: the sounding dataset that is read in 
            Eqn: Equation to use in the saturation vapor pressure calculation
            method: Noting which calculation you would like to use to dins the source origin height
            
        Outputs: 
            press_oforig: the pressure value of the source origin height in hPa
        """
        if method == 'DD_method1':
            '''this is the lowest thetae value in the sfc-400mb layer'''
            #only eval levels at which the pressure is greater than 400mb 
            sfc_to_400 = data.loc[data['press'] >= 400]
            #find the thetae-e at all the levels that matched the above criteria
            sfc_to_400_Te = thetae(sfc_to_400.press, sfc_to_400.tempK, Eqn) 
            #find which level the min thetae occurs at (from sfc to 400mb)
            row_oforig, Te_min = sfc_to_400_Te.idxmin(), sfc_to_400_Te.min()
            press_oforig = data.iloc[row_oforig]['press']
            return press_oforig

        
        elif method == 'DD_method2':   
            '''this is the lowest thetae value in 100mb averaged layers'''
            ##Group into 100 mb layers (label each group with number i in the colunmn layer_group)
            data['layer_group'] = np.nan
            i, start_p, end_p = 0, data['press'].iloc[0], data['press'].iloc[-1]
            
            while (start_p >= end_p):
                top_p = start_p - 100
                data.loc[(data['press'] <= start_p) & (data['press'] > top_p), 'layer_group'] = i
                i, start_p = i+1, top_p 

            #find the thetae-e at all the levels 
            data['Te'] = thetae(data.press, data.tempK, Eqn) 
            #Average the data via 100mb level groupings 
            data_averaged=data.groupby(pd.Grouper(key='layer_group')).mean()
            pressure_ave=data.groupby(pd.Grouper(key='layer_group')).median()
            #find which layer the min thetae occurs at 
            row_ofmin, Te_min = data_averaged['Te'].idxmin(), data_averaged['Te'].min()

            press_oforig = pressure_ave.loc[row_ofmin]['press']
            row_oforig = data.loc[data['press'] == press_oforig].index
            data=data.drop(columns=['layer_group'])
            return press_oforig

        elif method == 'DD_method3':   
            #Will be used to calculate the density weighted average DCAPE below
            press_oforig= data['press']   
            
            return press_oforig

    def DD_CAPE_CALC(data, sfc_press, upper_press, ES): 
        """ Using one of the options from above to calculate DCAPE
        ---
        Inputs: 
            data: the sounding dataset that is read in 
            sfc_press: surface pressure in hPa; lower bound of integration for DCAPE
            upper_press: upper level pressure in hPa; upper bound of integration for DCAPE
            ES: Noting which saturation vapor pressure equation you would like to use
            
        Output: 
            dcape: Downdraft CAPE in J/kg
        """
        # Trim data to only consider the levels within the identified layer
        # Flip order of the data to capture the descending motion of the parcel
        DD_layer = data.loc[(data['press'] <= sfc_press) & (data['press'] >= upper_press)].sort_values(by='press')
        ## Create the parcel profile for decent along a moist adiabat
        # # #  # # # #  # # #  # # # #  # # # # # #  # #  #  # #  # # 
        #calc parcel path temps (aka moist adiabtic descent) 
        parcel_temp = [DD_layer.tempK.values[0]]
        for i in range(1, len(DD_layer.index)):
            dz= DD_layer.hght[i]-DD_layer.hght[i-1] #new height - previous height
            new_temp=moist_lapse(parcel_temp[i-1], DD_layer.press.values[i-1], dz, ES)
            parcel_temp.append(new_temp)
        
        #convert to Celcius   
        pa_t=[x - 273.15 for x in parcel_temp]    
        #attach a new column of the parcel temps to the pandas dataframe
        DD_layer['p_tempC'], DD_layer['p_tempK'] = pa_t, parcel_temp
        DD_layer['TV_env'] = Virtual_Temp(DD_layer['press'], DD_layer['tempK'], ES)
        DD_layer['TV_par'] = Virtual_Temp(DD_layer['press'], DD_layer['p_tempK'], ES)
        ############
        
        ## Calculate the difference in profile and environmental temperature to integrate
        DD_layer['evn_par_diff']= DD_layer['TV_env'] - DD_layer['TV_par']
        with pd.option_context( 'display.max_columns', None):  # more options can be specified also
            DD_layer = DD_layer.drop(columns=['hght', 'dewC', 'dewK', 'tempK', 'p_tempK'])
            try: DD_layer = DD_layer.drop(columns=['layer_group'])
            except: pass
            #  print(DD_layer)
         
        # Calculate DCAPE
        dcape = ((mpconsts.Rd) * (np.trapz(DD_layer['evn_par_diff'], x=np.log(DD_layer['press'].values)) * units.kelvin)).to('J/kg')
        return dcape

    # Calculate bounds of integration
    # # # # # # #  # # # # #  # # # #
    sfc_press = data.iloc[0]['press'] #lower
    upper_press = DCAPE_origin(data, ES, origin_method) #upper
    
    #Option 3 is a density weighted average DCAPE
    if origin_method == 'DD_method3':
        dcape_array=[]
        for i in range(0, len(upper_press)):
            #Defining constants
            Rd = mpconsts.Rd.m*1000
            #Calculating the density of each temperature
            rho = ((data['press'][i] / (Rd*data['tempK'][i]))*-1000) #Converting from g to kg
            
            #Calculating the weighted average
            dcape_unweighted = DD_CAPE_CALC(data, sfc_press, upper_press[i], ES)
            dcape_weighted = dcape_unweighted * rho
            dcape_array.append(dcape_weighted)
            
        #Final_DCAPE = statistics.mean(dcape_array)
        Final_DCAPE = (sum(dcape_array)/(i+1))
   
    else: Final_DCAPE =DD_CAPE_CALC(data, sfc_press, upper_press, ES)
    
    calculation_time= time.time() - calc_start_time
    return Final_DCAPE.m, calculation_time

##########################################
#############
# Call Defns
#############
#Read in the requested sounding data 
sounding = readSounding(date, site)

#Create empty dictionarys that will hold our outputs
DD_outputs = {'DD_method1': { 'metpy':' ', 'Magnus': ' ', 'Wexler': ' ', 'Buck':' '},
              'DD_method2': {'metpy': ' ','Magnus': ' ', 'Wexler': ' ', 'Buck':' '},
              'DD_method3': {'metpy': ' ','Magnus': ' ', 'Wexler': ' ','Buck':' '}}
DD_times = {'DD_method1': { 'metpy':' ', 'Magnus': ' ', 'Wexler': ' ', 'Buck':' '},
            'DD_method2': {'metpy': ' ','Magnus': ' ', 'Wexler': ' ', 'Buck':' '},
            'DD_method3': {'metpy': ' ','Magnus': ' ', 'Wexler': ' ', 'Buck':' '}}

for origin_source, ES_versions in DD_outputs.items():
    for ES_eqn in ES_versions: 
        dcape_val, dcape_time = DCAPE(sounding, ES_eqn, origin_source)
        DD_outputs[origin_source][ES_eqn] = dcape_val 
        DD_times[origin_source][ES_eqn] = dcape_time
        #  print(origin_source, ES_eqn)
        #  print(DD_outputs[origin_source][ES_eqn], DD_times[origin_source][ES_eqn])

## #  # # # # # # # # # # # #  # #
###########
# Plotting
###########
DD_df, time_df = pd.DataFrame.from_dict(DD_outputs), pd.DataFrame.from_dict(DD_times)
print(DD_df)
labeling_dict = {'DD_method1':'Min Te <400mb', 'DD_method2':'Min Te 100mb layer', 'DD_method3':'Density Weighted'}
plt.rc('grid', color='w')

with plt.style.context('bmh'):
    ## PLOT A 3 Panel of DCAPE vs E_s method
    fig = plt.figure(figsize=(12,12))
    i=1
    for source_option, source_label in labeling_dict.items():
        ax = fig.add_subplot(3, 1, i)
        ax.plot(DD_df.index, DD_df[source_option].values)
        ax.set_ylabel(source_label)
        
        if i == 3: 
            ax.set_xlabel('E_s Eqn')
        elif i == 1: 
            ax.set_title('DCAPE Results Es Eqn and Downdraft Source Method Combos: '+site+' '+date, pad=20)
        i = i+1
    
    plt.tight_layout()
    plt_file = 'DCAPE_zoom'+'_'+site+'_'+date+'.png'
    fig.savefig(plt_file)
    plt.close()
    
    #  * * * * * * * * * * *

    ## PLOT all three DCAPE source method on the same chart (ie on the same scale)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(DD_df.index, DD_df.values)
    ax.set_title('DCAPE with various E_s Eqn and Source Origins: '+site+' '+date, pad=20)
    ax.set_ylabel('DCAPE (J/Kg)')
    ax.set_xlabel('E_s Eqn')
    
    #legend labels 
    lines= DD_df.columns 
    ax.legend(lines, prop={'size':10})

    plt.tight_layout()
    plt_file = 'DCAPE'+'_'+site+'_'+date+'.png'
    fig.savefig(plt_file)
    plt.close()

    #  * * * * * * * * * * *

    ## PLOT all the times needed for each DCAPE source/ E_s eqn combo)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)

    lines= time_df.columns 
    ax.plot(time_df.index, time_df.values)# , ylabel='Min Te <400mb')
    ax.set_title('Time Needed to Calculate DCAPE: '+site+' '+date, pad=20)
    ax.set_ylabel('Time (sec)')
    ax.set_xlabel('E_s Eqn')

    #legend labels 
    lines= time_df.columns 
    ax.legend(lines, prop={'size':10})
    plt.tight_layout()
    
    plt_file = 'Time'+'_'+site+'_'+date+'.png'
    fig.savefig(plt_file)
    plt.close()


# Summary of findings:
# Using the third method (density weighted average DCAPE) to find the source origin height seem to be slightly more accurate than the other methods using theta-e. The only issue is that this third method is slightly more computationally expensive than the other options, but it may be worth it for the accuracy. In terms of accuracy for the saturation vapor pressure calculations, it seems that the Magnus equation seem to produce the closest DCAPE value to the one respresented on the sounding. In this case, we used a sounding from ILX on August 10th, 2020 when the derecho impacted portions of the Midwest. DCAPE for 12 UTC on this date showed a value of 1490 J/kg. It seem thats the use of the lowest theta-e value in the sfc to 400 mb over predicted DCAPE and the use of theta-e from averaged 100 hpa layers under predicted the DCAPE values.
# 
# It would be beneficial to look at more calculations/equations of saturation vapor pressure to determine if there is an equation that produced more accurate DCAPE than what was found here. Also, we only considered theta-e related calculations with regard to finding the source origin height (the upper bound of the DCAPE integration). Within the Metpy Issue #823, it was suggested that wet-bulb temperature could also be used. 

# In[ ]:




