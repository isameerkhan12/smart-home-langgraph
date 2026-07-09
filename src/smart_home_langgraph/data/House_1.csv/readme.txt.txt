REFIT: Electrical Load Measurements

INFORMATION
Collection of this dataset was supported by the Engineering and Physical Sciences Research Council (EPSRC) via the project entitled Personalised Retrofit Decision Support Tools for UK Homes using Smart Home Technology (REFIT), which is a collaboration among the Universities of Strathclyde, Loughborough and East Anglia. The dataset includes data from 20 households from the Loughborough area over the period 2013 - 2015. Additional information about REFIT is available from www.refitsmarthomes.org.

LICENCING
This work is licensed under the Creative Commons Attribution 4.0 International Public License. See https://creativecommons.org/licenses/by/4.0/legalcode for further details.
Please cite the following paper if you use the dataset:

@inbook{278e1df91d22494f9be2adfca2559f92,
title = "A data management platform for personalised real-time energy feedback",
keywords = "smart homes, real-time energy, smart energy meter, energy consumption, Electrical engineering. Electronics Nuclear engineering, Electrical and Electronic Engineering",
author = "David Murray and Jing Liao and Lina Stankovic and Vladimir Stankovic and Richard Hauxwell-Baldwin and Charlie Wilson and Michael Coleman and Tom Kane and Steven Firth",
year = "2015",
booktitle = "Proceedings of the 8th International Conference on Energy Efficiency in Domestic Appliances and Lighting",
}

NAMING SCHEMEs
Each of the houses is labelled, House 1 - House 21 (skipping House 14), each house has 10 power sensors comprising a current clamp for the household aggregate and 9 Individual Appliance Monitors (IAMs). Only active power in Watts is collected at ~6-8 second interval.
The subset of all appliances in a household that was monitored reflects the document from DECC of the largest consumers in UK households, https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/274778/9_Domestic_appliances__cooking_and_cooling_equipment.pdf

FILE FORMAT
The file format is CSV (comma separated values) and is laid out as follows;
DateTime (YYYY-MM-DD HH:mm:ss), UNIX TIMESTAMP (UCT), Aggregate, Appliance1, Appliance2, Appliance3, ... , Appliance9
Additionally data was only recorded when there was a change in load; this data has been forward filled with the last actual value recored unless the gap is greater than 2 minutes in which case it was filled with 0. The sensors are also not synchronised as our collection script polled every 6-8 seconds; the sensor may have updated anywhere in the last 6-8 seconds.
The CSV files have more rows than Excel is able to display at one time and should therefore be opened using a program such as Matlab or other data processing application.

MISSING DATA
During the course of the study there are a few periods of missing data (notably February 2014). Outages were due to a number of factors, including household internet failure, hardware failures, network routing issues.

DATA CLEANING
All of the data has been cleaned to forward fill NaN values and remove values which exceed 4000 Watts on the IAM datastreams as this is above the maximum possible power draw.

APPLIANCE LIST
The following list shows the appliances that were known to be monitored at the beginning of the study period. Although occupants were asked not to remove or switch appliances monitored by the IAMs, we cannot guarantee this to be the case. It should also be noted that Television and Computer Site's may consist of multiple appliances, e.g. Television, SkyBox, DvD Player, Computer, Speakers, etc.
Where Computer/Television Site's have the list of appliances known these are located in the notes section for that house. The notes section also contains additional information about additional appliances or changes which may have occurred. Please make sure to read this as it highlights important changes.

House 1
0.Aggregate, 1.Fridge, 2.Chest Freezer, 3.Upright Freezer, 4.Tumble Dryer,
5.Washing Machine, 6.Dishwasher, 7.Computer Site, 8.Television Site, 9.Electric Heater
	!NOTES
		0. October 2014, numerous light bulbs changed to LEDs.
		7.a Desktop Computer
		7.b Computer Monitor

