# MarsLab Feature Discussion: Mission Planning and Data Analysis

## PARTICIPANTS
*   **Dr. Elena Vasquez**: Mars Geologist / Principal Investigator
*   **James Park**: Project Lead / Systems Engineer
*   **Dr. Anika Rao**: Rover Software Engineer
*   **Marcus Chen**: Ground Operations Lead
*   **Dr. Fatima Al-Rashid**: Spectral Data Scientist

---

## Part 1: Mission Planning Phase

**James Park**: Alright everyone, let's get started. We have the landing site coordinates for the next phase of the MarsLab mission, but we need to finalize the traverse path for the rover. Elena, you had some specific targets in mind for the mineral analysis?

**Dr. Elena Vasquez**: I do. I've been looking at the regional context, and there's a specific outcrop near the western rim of the crater that looks promising for phyllosilicates. Marcus, can you pull up the MapView and center it on our current coordinates?

**Marcus Chen**: Doing that now. I'm using the TopBar search to jump to the site. I'll just type in the coordinates... there we go. I'm switching the base layer to the HRSC terrain so we can see the topography better.

**Dr. Elena Vasquez**: Thanks, Marcus. Now, open the LayerPanel and toggle the CRISM footprints. I want to see what hyperspectral coverage we have for that outcrop.

**Marcus Chen**: Loading the Spectral group. The cyan polygons for the CRISM MTRDR products are appearing. It looks like we have three overlapping observations right on that western rim.

**Dr. Anika Rao**: Wait, before we get too excited about the science, I need to check the terrain. Marcus, can you enable the DTMHoverReadout? I want to see the elevation changes as you move the cursor along that proposed path.

**Marcus Chen**: Sure thing, Anika. I'm moving the mouse over the ridge now. The readout is showing a pretty steep drop.

**Dr. Anika Rao**: Yeah, that looks like a thirty degree slope in some spots. The rover's autonomy system is going to flag that. We might need to find a more gradual approach. Can you open the SlopeAnalysis3DTab for that specific area?

**Marcus Chen**: I'll need to load the HiRISE DTM first. I'm selecting the brown footprint in the LayerPanel. Okay, the DTM is loaded. I'm opening the SlopeAnalysis3DTab now.

**Dr. Anika Rao**: That's better. The heatmap shows a lot of red along the direct route Elena wants. If we swing about two hundred meters to the south, it looks like there's a natural ramp with slopes under fifteen degrees.

**Dr. Elena Vasquez**: If we go that far south, do we still have CRISM coverage? I don't want to spend three sols driving if we're moving away from the primary spectral targets.

**Dr. Fatima Al-Rashid**: I'm looking at the footprints now. If we take Anika's southern route, we're still within the footprint of the MTRDR0000B462 observation. I can pull up the Inspector for that product to see if the signal to noise ratio is good enough in that southern section.

**Marcus Chen**: I've clicked the footprint. The Inspector is opening on the right.

**Dr. Fatima Al-Rashid**: Elena, look at the spectral plot in the Inspector. I'm selecting the bands for the 1.9 and 2.3 micron features. The absorption looks deep even on the periphery of the image. I think the southern route is viable for science.

**James Park**: Good. Marcus, use the TimelineNavigator to see if there are any more recent CTX images of that southern ramp. I want to make sure there aren't any new boulder fields that the older HiRISE data might have missed.

**Marcus Chen**: I'm sliding the timeline to the most recent date. There's a CTX image from last month. The pink footprint is showing up. It's a bit grainy compared to HiRISE, but the ramp looks clear of major obstacles.

**Dr. Elena Vasquez**: While you're there, Marcus, can you use the MeasurementTools to give me a rough distance for that southern detour?

**Marcus Chen**: I'm clicking the ruler icon. Point A at the current location, Point B at the ramp, Point C at the outcrop. It's about four hundred and fifty meters total.

**James Park**: That's manageable. We can do that in two sols if the autonomy holds up. Anika, what do you think?

**Dr. Anika Rao**: If the terrain is as smooth as the CTX context suggests, we can probably push the speed a bit. I'll want to run a more detailed simulation once we have the full HiRISE DTM patch extracted.

**Dr. Elena Vasquez**: One more thing. I noticed a small feature in the SHARAD ground tracks nearby. Marcus, can you toggle the SHARAD group in the LayerPanel?

**Marcus Chen**: Loading the orange LineString tracks. There's one that passes right through the crater floor, just east of our outcrop.

**Dr. Elena Vasquez**: Can we see the radargram for that? I'm curious if there's any subsurface layering that might indicate a buried delta structure.

**Marcus Chen**: I'm clicking the track to open the SharadHiresInspector. The radargram is loading. I'll turn on the MOLA surface profile overlay so we can distinguish the surface return from the subsurface echoes.

**Dr. Fatima Al-Rashid**: There's a definite reflector about fifty meters down. It's quite sharp.

**Dr. Elena Vasquez**: That's interesting. If that's a contact between two different units, it might change our priority for the outcrop. We might want to spend more time at the base of the ridge where that layer might be exposed.

**James Park**: This is exactly why we use the integrated view. Marcus, save this traverse plan. We'll need to present it to the mission board tomorrow.

**Marcus Chen**: I'll create a FieldNoteModal at the outcrop location and attach the traverse coordinates. I'm tagging it as "Priority 1 Science" and "Traverse Plan Alpha".

---

## Part 2: Data Collection & Analysis Phase

**James Park**: We just got the downlink from Sol 42. The rover reached the outcrop and performed a series of spectral scans. Marcus, have the data products been ingested into the pipeline?

**Marcus Chen**: Yes, the CRISM TRR3 data is ready. I've already updated the LayerPanel. You should see the new footprints in yellow.

**Dr. Elena Vasquez**: Excellent. Let's see what we've got. Marcus, open the Inspector for the new TRR3 observation.

**Marcus Chen**: Opening it now. I'm focusing on the area where the rover is currently parked.

**Dr. Fatima Al-Rashid**: I've already started running the CNN classification on this dataset. I'm going to open the MineralSequencePanel to see if the model has identified any specific trends in the mineralogy as we moved up the section.

**Dr. Elena Vasquez**: What's the model saying, Fatima?

**Dr. Fatima Al-Rashid**: It's flagging a high confidence for magnesium-rich olivine at the base, transitioning into more hydrated minerals as we go higher. But I want to verify this manually. Marcus, can you open the BandRatioCalculator?

**Marcus Chen**: Sure. Which indices do you want to run?

**Dr. Fatima Al-Rashid**: Let's start with the olivine index. Use the standard bands for the 1.0 and 1.5 micron ratios.

**Marcus Chen**: Calculating... the score map is loading over the MapView. It's showing a lot of purple and blue at the base of the outcrop.

**Dr. Elena Vasquez**: That matches the CNN prediction. Now, let's look at the hydration features. Run the BD1900 index.

**Marcus Chen**: Running the 1.9 micron band depth calculation. Okay, that's interesting. The hydration signal is much stronger in the upper layers, just like Fatima's model suggested.

**Dr. Fatima Al-Rashid**: I'm a bit concerned about the noise in the TRR3 data, though. The atmospheric correction might be overcompensating in some of the bands. Elena, can we compare this with the MTRDR data we looked at during planning?

**Dr. Elena Vasquez**: Good idea. Marcus, use the SpectralComparison tool. I want to see the TRR3 spectrum from the rover's current location side-by-side with the MTRDR spectrum from the same spot.

**Marcus Chen**: I'm selecting both products and entering ComparisonMode. I've got the two spectral plots up now.

**Dr. Fatima Al-Rashid**: Look at the 2.1 micron region. There's a slight shift in the TRR3 data that isn't in the MTRDR. That could be a calibration issue, or it could be a real mineralogical difference that the lower-resolution MTRDR missed.

**Dr. Elena Vasquez**: Or it could be dust. Anika, did the rover's cameras show any significant dust accumulation on the optics?

**Dr. Anika Rao**: The health checks were all green, but we did have a small wind event on Sol 40. I can check the onboard logs. But honestly, the AiAnalysisPanel might be faster. I'll ask it to look for evidence of dust interference in the recent spectral acquisitions.

**Marcus Chen**: I'm opening the AiAnalysisPanel. Anika, do you want to run the query?

**Dr. Anika Rao**: Yeah. "Analyze recent CRISM TRR3 spectra for potential dust contamination or atmospheric artifacts based on Sol 40 weather data."

**Marcus Chen**: The agent is starting the investigation. It's pulling data from the weather sensors and the spectral pipeline.

**Dr. Fatima Al-Rashid**: While that's running, I'm going to use the MineralSequencePanel to build a stratigraphic column. I want to see how these mineral changes align with the elevation data from the HiRISE DTM.

**Marcus Chen**: I'll open the StratColumnPanel for you. I'm linking it to the DTM so the elevations are accurate.

**Dr. Fatima Al-Rashid**: Thanks. I'm adding the olivine unit at the bottom, then the smectite layer above it. The transition seems to happen right at the three hundred and forty meter elevation mark.

**Dr. Elena Vasquez**: That's a very sharp contact. If that's a global marker, it could be huge for our understanding of the crater's history.

**Marcus Chen**: The AiAnalysisPanel just finished. It says the spectral shift at 2.1 microns is likely not dust. It's suggesting a possible match for a rare carbonate species that wasn't in our initial library.

**Dr. Elena Vasquez**: Carbonates? In this environment? That would change everything. We need to be absolutely sure.

**Dr. Fatima Al-Rashid**: I'll run a more detailed hyperspectral analysis. Marcus, can you export the raw spectral data for that pixel to a CSV? I want to run it through my local ML models.

**Marcus Chen**: I'm on it. I'll use the DataDownloadPage to queue the export.

---

## Part 3: Collaboration & Decision-Making Phase

**James Park**: Okay, we have a situation. The potential carbonate signature Elena and Fatima found is causing a lot of excitement, but we only have two sols left in this region before we have to move towards the winter haven. We need to decide if we're going to stay and investigate or stick to the original plan.

**Dr. Elena Vasquez**: We have to stay. If this is a carbonate, it's a direct indicator of a past neutral-pH aqueous environment. It's a high-priority science target.

**Dr. Anika Rao**: But the winter haven isn't optional, Elena. If we don't get the rover to the north-facing slope by Sol 60, we won't have enough power to keep the heaters running through the night. We're already cutting it close.

**James Park**: Let's look at the options. Marcus, can you open the ComparisonTray? I want to see the current location versus the winter haven route.

**Marcus Chen**: I'm pulling up the MapView. I've got the current outcrop on the left and the proposed traverse to the north on the right.

**Dr. Elena Vasquez**: What if we use the AgenticPanel to run a multi-step investigation? Maybe it can find a way to confirm the carbonate signature without a full three-sol stay.

**Marcus Chen**: I'll start a new session in the AgenticPanel. "Conduct a multi-instrument investigation of the potential carbonate signature at the current location. Prioritize speed and identify the minimum set of observations needed for high-confidence confirmation."

**Dr. Fatima Al-Rashid**: While the agent is working, I'm looking at the SHARAD data again. Marcus, can you bring up the Subsurface3DViewer? I want to see if that reflector we saw earlier has any relationship to this carbonate layer.

**Marcus Chen**: Loading the 3D subsurface view. I'm overlaying the mineral map from Fatima's CNN classification on top of the radargram.

**Dr. Fatima Al-Rashid**: Look at that. The carbonate signature seems to be coming from the same stratigraphic level as the SHARAD reflector. That suggests the layer isn't just a surface coating; it's a substantial unit that extends into the subsurface.

**Dr. Elena Vasquez**: That's a massive piece of evidence. If it's a continuous layer, we don't need to stay at this exact spot. We might be able to find another exposure further along our path to the winter haven.

**Dr. Anika Rao**: Marcus, can you use the Proximity search to find any other outcrops with similar spectral signatures along the northern traverse?

**Marcus Chen**: I'm using the TopBar to run a spatial search. "Find all CRISM products within five kilometers of the northern traverse path that show high BD1900 and carbonate indices."

**Marcus Chen**: The results are coming in. There's a small cluster of footprints about two kilometers to the north. It's right on our way.

**Dr. Elena Vasquez**: Let me see the Inspector for those products.

**Marcus Chen**: Opening the Inspector for MTRDR0000C124.

**Dr. Elena Vasquez**: The signal is weaker there, but the features are definitely present. It's not as good as our current spot, but it might be enough.

**James Park**: The AgenticPanel just finished its report. It's suggesting a single high-resolution spectral scan and a series of close-up images with the rover's microscopic imager. It estimates this will take about six hours.

**Dr. Anika Rao**: Six hours we can do. If we start now, we can still finish the drive to the next waypoint by tomorrow evening.

**Dr. Elena Vasquez**: I'm okay with that. But I want to make sure we document this decision thoroughly. Marcus, can you generate a report based on this session?

**Marcus Chen**: I'm opening the ReportPanel. I'll include the spectral plots, the SHARAD 3D view, and the AgenticPanel's recommendations. I'm also adding the citations for the carbonate ML model.

**Dr. Fatima Al-Rashid**: Make sure to include the comparison between the current site and the northern site. We need to show why we're confident we can find the same material later.

**Marcus Chen**: I'm adding the ComparisonMode screenshots to the report. The AI is generating the summary now.

**James Park**: Once the report is ready, I'll send it to the ground stations for the final uplink. Anika, start prepping the command sequences for the microscopic imager.

**Dr. Anika Rao**: Already on it. I'm using the GuidedWorkflows to make sure I don't miss any of the instrument calibration steps.

**Dr. Elena Vasquez**: This is a good compromise. We get the data we need without risking the rover.

**Marcus Chen**: The report is finished. I'm sharing the link with the whole team. I've also added a FieldNoteModal at the new northern target so we don't forget to look for it when we get there.

**James Park**: Great work, everyone. This is exactly how the MarsLab system is supposed to work. We found an anomaly, analyzed it with multiple instruments, used the AI to weigh our options, and made a data-driven decision in under two hours.

---

## Part 4: Feature Derivation Summary

Based on the team's discussion and the challenges they faced during the mission planning and analysis phases, the following features have been prioritized for the MarsLab platform.

### Core Features (MVP)
These features are essential for basic mission operations and were used constantly throughout the discussion.

1.  **MapView with Multi-Layer Support**
    *   **Description**: A 3D Cesium-based globe that supports multiple base layers (MOLA, HRSC) and instrument footprints.
    *   **Roles**: All roles.
    *   **User Story**: Marcus needs to toggle between different terrain and spectral layers to provide context for Elena's science targets.

2.  **Instrument Inspector**
    *   **Description**: A detailed panel for inspecting individual data products, including metadata, spectral plots, and band selection.
    *   **Roles**: Elena, Fatima.
    *   **User Story**: Elena uses the Inspector to examine the spectral signatures of specific pixels to identify minerals.

3.  **LayerPanel with Hierarchical Organization**
    *   **Description**: A tool for managing the visibility of various instrument groups (Spectral, Imagery, Radar).
    *   **Roles**: Marcus, Elena.
    *   **User Story**: Marcus organizes the map by toggling CRISM and SHARAD layers to reduce visual clutter during planning.

4.  **Spatial and Natural Language Search**
    *   **Description**: A search bar that supports both coordinate-based jumps and AI-powered natural language queries.
    *   **Roles**: Marcus, Anika.
    *   **User Story**: Anika uses the search to find regions with specific slope characteristics for rover safety.

5.  **FieldNoteModal**
    *   **Description**: A tool for creating geolocated notes with tags and attachments to document decisions and findings.
    *   **Roles**: James, Marcus.
    *   **User Story**: James insists on documenting the traverse plan and the carbonate discovery for the mission board.

### High-Priority Features (Next Iteration)
These features provide advanced analysis capabilities that were critical for resolving the "carbonate anomaly" scenario.

1.  **AgenticPanel for Multi-Step Investigation**
    *   **Description**: An AI-driven tool that can perform complex, multi-instrument analysis workflows and provide recommendations.
    *   **Roles**: Elena, James, Anika.
    *   **User Story**: The team uses the AgenticPanel to determine the most efficient way to confirm a new mineral discovery under time pressure.

2.  **SharadHiresInspector and Subsurface3DViewer**
    *   **Description**: Specialized tools for visualizing and analyzing SHARAD radar data in both 2D radargrams and 3D volumes.
    *   **Roles**: Elena, Fatima.
    *   **User Story**: Fatima correlates a subsurface radar reflector with a surface mineral signature to confirm the thickness of a geological unit.

3.  **BandRatioCalculator and MineralSequencePanel**
    *   **Description**: Tools for performing mathematical operations on spectral bands and visualizing mineralogical trends over time or elevation.
    *   **Roles**: Fatima, Elena.
    *   **User Story**: Fatima runs an olivine index to verify the CNN's mineral classification at the base of the outcrop.

4.  **ComparisonMode and ComparisonTray**
    *   **Description**: A split-screen interface for comparing two different data products or locations side-by-side.
    *   **Roles**: Fatima, James.
    *   **User Story**: Fatima compares TRR3 and MTRDR spectra to identify potential calibration artifacts.

5.  **ReportPanel**
    *   **Description**: An AI-assisted tool for generating comprehensive geological reports with embedded data visualizations and citations.
    *   **Roles**: James, Marcus.
    *   **User Story**: Marcus generates a formal report to justify the change in the mission plan to the ground operations team.

### Nice-to-Have Features (Future Roadmap)
These features enhance the user experience and provide additional context but are not strictly necessary for the core mission workflow.

1.  **TimelineNavigator**
    *   **Description**: A temporal slider for navigating data products based on their acquisition date.
    *   **Roles**: James, Marcus.
    *   **User Story**: James uses the timeline to find the most recent CTX images to check for new obstacles.

2.  **SlopeAnalysis3DTab**
    *   **Description**: A specialized view for analyzing terrain slopes in 3D to ensure rover safety.
    *   **Roles**: Anika.
    *   **User Story**: Anika identifies a safe "ramp" for the rover by analyzing the slope heatmap on a HiRISE DTM.

3.  **MeasurementTools**
    *   **Description**: Tools for measuring distances, areas, and elevation profiles directly on the map.
    *   **Roles**: Marcus, Elena.
    *   **User Story**: Marcus calculates the length of a proposed traverse detour to estimate the time required.

4.  **OnboardingTour**
    *   **Description**: A guided walkthrough for new users to learn the various tools and panels in MarsLab.
    *   **Roles**: New team members.
    *   **User Story**: A new intern uses the tour to understand how to load CRISM data for the first time.

5.  **SpaceGame (Easter Egg)**
    *   **Description**: A hidden mini-game for team members to play during long data downlink periods.
    *   **Roles**: All roles.
    *   **User Story**: The team plays a quick game while waiting for the large HiRISE DTM files to finish loading.

### Identified Gaps and Future Needs
During the discussion, the team identified several areas where the current MarsLab features could be improved:

*   **Inline Confidence Scores**: Fatima noted that the Inspector should show the CNN's confidence scores directly next to the mineral identifications.
*   **Real-Time Cursor Collaboration**: James mentioned that it would be helpful to see where other team members are looking on the map during the meeting.
*   **Automated Dust Detection**: Anika suggested that the spectral pipeline should automatically flag potential dust interference based on weather data.
*   **Direct DTM to SHARAD Correlation**: Elena wanted a more seamless way to overlay SHARAD reflectors onto the 3D terrain view to see where they might out crop.
