# MarsLab Discussion 2: Subsurface Science in Arcadia Planitia

**Date:** February 25, 2026  
**Participants:**  
- **Dr. Elena Vasquez** — Mars Geologist / Principal Investigator  
- **James Park** — Project Lead / Systems Engineer  
- **Dr. Anika Rao** — Rover Software Engineer  
- **Marcus Chen** — Ground Operations Lead  
- **Dr. Fatima Al-Rashid** — Spectral Data Scientist  
- **Dr. Yuri Petrov** — Planetary Geophysicist / Radar Science Lead  

---

## Part 1: Arcadia Planitia Traverse Context

**Elena:** Let's get started. We're looking at the northern plains today. Specifically, the region between 40 and 50 degrees North, 180 to 200 East. James, can you pull up the Arcadia Planitia base map in the MapView? We need to establish the regional context before we zoom into the high resolution targets.

**James:** Loading it now. I've got the MOLA colorized topography as the base layer. I'm also toggling on the HiRise footprint layer so we can see where we have high resolution coverage. The MapView is refreshing. You should see the global mosaic now, with the Arcadia region centered.

**Marcus:** It looks remarkably flat. That's good for the rover's power budget, but I'm worried about the lack of landmarks for visual odometry. If we're driving for kilometers across a featureless plain, we'll have to rely heavily on the IMU and wheel encoders.

**Anika:** We can handle the odometry, Marcus. My concern is the regolith consistency. If it's all fine grained dust, we'll have traction issues. We've seen what happens when rovers get bogged down in soft drifts. I'm looking at the DTMHoverReadout values as I move the cursor. The slopes are negligible, but the surface texture in the HiRise stamps looks suspiciously uniform.

**Yuri:** It's not just dust, Anika. That's the mistake people made for decades. They looked at the surface and saw a desert. But the radar tells a different story. Elena, if you open the LayerPanel and toggle the SHARAD ground tracks, you'll see why this area is so special.

**Elena:** Done. The map is covered in those purple lines now. Yuri, you've been looking at the latest Bramson data, right? I remember the buzz when that paper first came out.

**Yuri:** Exactly. Bramson et al. (2015) published a critical paper in GRL titled "Widespread excess ice in Arcadia Planitia, Mars." They used SHARAD data to identify a massive subsurface ice sheet. It's not just a few pockets or isolated glaciers. We're talking about ice that extends from just a few meters below the surface down to depths of over 100 meters. The sheer scale of it is staggering. It covers hundreds of thousands of square kilometers.

**Fatima:** I'm looking at the Inspector panel for the coordinates Yuri mentioned. The thermal inertia values are interesting. They don't quite match pure basaltic regolith. If I compare them to the global averages in the SpectralComparison tool, there's a clear deviation.

**Yuri:** They shouldn't match. Bramson's team found dielectric constants, or epsilon r, between 3 and 4. For those who don't live in the radar world, pure water ice is about 3.15. Basaltic rock is usually 4 to 8. A value of 3 to 4 means we're looking at ice rich material, possibly nearly pure ice with a thin lag deposit on top. The Bramson paper used the delay between the surface and the subsurface reflector to map the thickness of this ice sheet across the entire basin.

**James:** So the rover isn't just driving on dirt. It's driving on a glacier with a dusty blanket? That changes the engineering requirements for the wheels and the thermal system.

**Yuri:** In many places, yes. If you look at the SHARAD tracks in the MapView, you can see the reflectors are very strong. This isn't a subtle signal that you have to squint to see. The Bramson paper suggests this ice is widespread across the mid latitudes of Arcadia. They used a two layer model to estimate the thickness. The top layer is the regolith, and the second layer is the ice.

**Elena:** This changes our traverse priorities. We aren't just looking for interesting rocks anymore. We're looking for access points to that ice. If we can sample it, we're looking at a record of the Martian climate from millions of years ago.

**Marcus:** Access points? You mean where the ice is closest to the surface? We need to find a spot where the rover can actually reach it without digging a massive trench.

**Yuri:** Precisely. We need to find areas where the "overburden," the regolith on top, is thin enough for our drill or for natural exposure. If we can find a spot where the ice is within the top two meters, that's a primary science target. The Bramson paper mapped the overburden thickness, and it varies quite a bit. Some areas have tens of meters of dust, but others are much thinner.

**Anika:** I'm looking at the SlopeAnalysis3DTab. If we're going to target ice, we need to be careful about the slopes. Ice rich ground can be unstable if it's on a steep incline, though Arcadia is mostly flat. I'm also worried about the bearing strength. If the ice is shallow, the ground might be harder than we expect, or it might be more brittle.

**Fatima:** I've loaded the CRISM stamps for this area. I'm using the BandRatioCalculator to look for the 1.5 and 2.0 micron water ice absorption features. It's tough because the dust cover is so pervasive. The dust masks the spectral signature of the ice underneath. I'm trying to find "windows" where the dust has been cleared by recent impacts or wind.

**Yuri:** Don't rely solely on the surface spectral data, Fatima. The radar sees what the spectrometer misses. The 20 MHz signal from SHARAD goes right through that dust like it's not even there. Let's dive into the actual radargrams. James, can you open the SharadHiresInspector for track 12345_01? It cuts right through our primary candidate site near 45 North.

**James:** Opening SharadHiresInspector. I've got the radargram on the left and the MOLA surface profile on the right. I'm syncing the cursor so we can see exactly where we are on the map as we scroll through the radar data.

---

## Part 2: SHARAD Analysis Deep Dive

**Yuri:** Perfect. Now, look at that secondary reflector about 0.5 microseconds below the surface return. Elena, do you see that sharp, continuous line? It's almost as bright as the surface itself.

**Elena:** I see it. It follows the surface topography almost perfectly. That usually suggests a compositional boundary, not just a random rock layer. If it were a geological contact between two different types of rock, it would likely be more irregular.

**Yuri:** Correct. This is where we need to be rigorous. We can't just guess the depth. We have to calculate it using the two way travel time. Seu et al. (2007) laid out the SHARAD instrument specs in their JGR paper. The center frequency is 20 MHz with a 10 MHz bandwidth. That gives us a range resolution of about 15 meters in free space. But we have to account for the medium.

**Anika:** But we aren't in free space. We're in the Martian subsurface. The waves slow down, right?

**Yuri:** Exactly. The resolution improves as the wave slows down in the medium. The depth, d, is equal to the time delay times the speed of light, divided by two times the square root of the dielectric constant. The square root of epsilon r is the refractive index of the material. For ice, it's about 1.78.

**James:** I'll pull up the MeasurementTools. If I click the surface and then the reflector, the tool gives me a delta t of 0.45 microseconds. I'm using the crosshair to be as precise as possible.

**Yuri:** Okay, let's do the math. If we assume the material is pure ice, we use an epsilon r of 3.15. That gives us a depth of about 38 meters. But if it's a mix of ice and regolith, say an epsilon r of 5, the depth drops to about 30 meters. This is the fundamental ambiguity in radar sounding. You can't know the depth unless you know the material, and you can't know the material unless you know the depth.

**Elena:** This is the debate we always have. Is it pure ice or ice rich regolith? The difference is huge for our science goals. Pure ice means a massive reservoir of water. Ice rich regolith might just be frozen soil.

**Yuri:** We can look at the work by Petersen et al. (2018). Their paper, "All Our Aprons Are Icy," focused on Deuteronilus Mensae, but the methodology is identical. They used SHARAD data and MOLA thickness measurements to calculate the dielectric constant directly. They found values of 3.0 to 3.15 for those lobate debris aprons. That's almost pure water ice. They argued that if the dielectric constant is that low, there's simply no room for much rock or dust in the mixture.

**Fatima:** Could we apply that same logic here? If we have an independent measure of the layer thickness, we could solve for epsilon r. Maybe from a nearby crater that penetrates the layer?

**Yuri:** We're trying. In Arcadia, we don't have as many clear "bottom" reflectors where the ice hits the bedrock, but where we do, the numbers keep coming back low. It's ice, Elena. It's not just "damp" soil. It's a buried glacier. The Bramson paper also looked at the loss tangent. Ice is very transparent to radar. If the material were mostly rock, the signal would fade much faster.

**Elena:** If it's that pure, the loss tangent must be very low. That explains why the reflectors are so clear even at depth.

**Yuri:** It is. The signal penetrates deep without much attenuation. That's why we see such clear reflectors even at 100 meters depth. If this were solid basalt, the signal would be scrambled or absorbed much faster. The Seu et al. (2007) paper explains how the SHARAD signal is processed to maximize the signal to noise ratio. We're seeing the benefit of that processing here.

**James:** I'm using the depth conversion feature in the SharadHiresInspector now. I've set the epsilon r to 3.15 as Yuri suggested. The tool is re-scaling the Y axis from microseconds to meters. Now we can see the actual thickness of the overburden.

**Marcus:** That reflector is deep. 40 meters is way beyond our drill's reach. We need something shallower. Our current drill is only rated for two meters. We'd be spinning our wheels for nothing at this location.

**Yuri:** Look further down the track, near the 45 degree North mark. See how the reflector rises? It almost merges with the surface return. The delay is shrinking as we move north.

**Anika:** I see it. The delay there is less than 0.1 microseconds. That's right at the limit of SHARAD's resolution. If the delay is smaller than the pulse width, the two reflections start to overlap.

**Yuri:** That's our target. At that location, the ice might be within the top 5 to 10 meters. We might even find surface expressions if we look at the HiRise data. The Bramson paper identified several of these "shallow ice" zones. They're often associated with specific surface textures.

**Elena:** Let's switch to the HiRiseDTM3DViewer for that specific coordinate. I want to see the surface texture. If there's ice that close, we might see patterned ground or thermal contraction cracks. James, can you load the DTM for the 45.2 North site?

---

## Part 3: Science-Engineering Integration

**James:** Switching to HiRiseDTM3DViewer. I'm centering on 45.2N, 190.5E. The 25 centimeter per pixel resolution is coming through now. I'm also loading the SlopeAnalysis3DTab so we can see the gradients in real time.

**Elena:** Look at those polygons. That's classic periglacial terrain. It's almost identical to what we see in the Canadian Arctic or the Dry Valleys of Antarctica. Those polygons form when the ground freezes and contracts, creating a network of cracks that often fill with ice.

**Anika:** The slopes look manageable. I'm checking the SlopeAnalysis3DTab. Most of the area is under 5 degrees. That's well within the rover's safety limits. But wait, look at that scarp to the east. It's a sharp drop.

**Yuri:** That's exactly what I was hoping for. Anika, you've found one of the sites discussed in Dundas et al. (2018). That Science paper, "Exposed subsurface ice sheets in the Martian mid-latitudes," is a game changer for traverse planning. They used HiRise to find places where the ice is actually visible.

**Elena:** I remember that paper. They found eight sites where steep scarps expose nearly pure water ice. They used the color data from HiRise to confirm the presence of ice.

**Yuri:** Yes, and several are in this region. The ice in those scarps starts just one or two meters below the surface and extends down over 100 meters. It's blueish in the enhanced color HiRise images. The Dundas paper shows that this ice is massive and relatively free of debris. It's not just a thin layer; it's a huge slab.

**Marcus:** A 100 meter ice cliff? That sounds like a nightmare for rover safety. We can't get too close to a crumbling scarp. If a chunk of that ice falls, it could crush the rover. We need to maintain a safe standoff distance.

**James:** We don't need to climb it, Marcus. We just need to get the rover's instruments within range of the base or the top edge. We could use the remote sensing instruments, like the ChemCam or the SuperCam, to analyze the ice from a distance.

**Anika:** If the rover drives on that exposed ice, what happens to our traction? Our wheels are designed for regolith. Metal on ice has a very low coefficient of friction. We could slide right off the edge if we aren't careful. I'll need to update the path planning algorithms to avoid any areas with exposed ice.

**Yuri:** We'd need to stay on the regolith covered parts. But the proximity to the ice is what matters. We can use the rover's RIMFAX or a similar ground penetrating radar to confirm what SHARAD sees at a much higher resolution. RIMFAX can see the top few meters in incredible detail. It would bridge the gap between the surface and the SHARAD data.

**Fatima:** I'm running the CNN Mineral Classification on the HiRise and CRISM data for that scarp. The model is flagging high probabilities for hydrated minerals along the lower layers. I'm seeing signatures that look like gypsum or other sulfates.

**Elena:** That makes sense. As the ice sublimates, it leaves behind whatever was trapped in it. We might be looking at a concentrated record of Martian climate history. The minerals could tell us about the chemistry of the water that formed the ice.

**Fatima:** I'll use the SpectralComparison tool to compare the scarp's signature with the surrounding plains. The difference is stark. The scarp has a much higher albedo and a distinct water ice absorption feature that isn't just surface frost. The 1.5 micron band is very deep here.

**Yuri:** This is where the Bramson and Dundas papers converge. Bramson shows the ice is everywhere, and Dundas shows us exactly what it looks like in cross section. It's massive, relatively pure, and accessible. The Dundas paper even estimated the purity of the ice by looking at the sublimation rates. They concluded it's over 90 percent water ice.

**James:** I'm looking at the Subsurface3DViewer. I've imported the SHARAD reflectors as 3D surfaces and overlaid the HiRise DTM. You can see the ice table sitting right under the surface like a hidden floor. It's like we're looking through the ground.

**Anika:** It's beautiful, but it's a thermal challenge. Ice has a much higher thermal conductivity than porous regolith. The rover's electronics might lose heat faster if we're parked over shallow ice. We'll need to keep the heaters running longer, which drains the battery.

**Marcus:** We'll need to adjust the sleep cycles. And we'll need to be careful with the drill. If we generate too much heat, we could melt the ice and get the bit stuck in a refrozen mess. We've seen that happen in terrestrial polar drilling.

**Yuri:** That's a real risk. But the science payoff is worth it. We're talking about direct access to the Martian hydrosphere. We could look for organic molecules trapped in the ice.

---

## Part 4: Synthesis and MarsLab Improvement Ideas

**Elena:** This has been an incredible session. We've gone from a flat, boring plain to a complex glacial landscape. Yuri, your input on the radar data has been vital. It's clear that we need to integrate these different data types more closely.

**Yuri:** I'm glad I could help. But looking at how we're working, I see some ways we could make MarsLab even better for this kind of analysis. We're doing a lot of manual steps that could be automated.

**James:** I'm all ears. What's on your list? We're always looking for ways to improve the workflow.

**Yuri:** First, we need an integrated dielectric constant calculator in the SharadHiresInspector. Right now, we're doing the math on the side or manually changing settings. I want a tool where I can click a reflector, input a known depth from a scarp or a DTM, and have it calculate the epsilon r for me instantly. It should also show the uncertainty based on the range resolution.

**Anika:** That's easy to implement. We can add a "Dielectric Estimator" mode to the MeasurementTools. We could even have it suggest values based on the literature, like the Petersen et al. (2018) results.

**Yuri:** Second, we should combine the thermal models with the radar data. If we know the depth of the ice from SHARAD, we can calculate the ice stability depth more accurately. We could create an "Ice Stability Overlay" in the LayerPanel that shows where the ice is currently sublimating versus where it's protected. This would be invaluable for choosing landing sites.

**Fatima:** I'd love to see an automated reflector to surface outcrop correlation. If SHARAD shows a reflector rising toward the surface, MarsLab should automatically highlight any nearby scarps or polygons in the HiRise data. It could use the CNN to look for those specific textures.

**Marcus:** From an ops perspective, I want a "Traverse Hazard Score" that includes subsurface ice proximity. If we're over shallow ice, the score should reflect the thermal and traction risks Anika mentioned. We could color code the traverse path based on the risk level.

**Elena:** And for validation, we should be able to import the published data directly. I want to see Bramson's original thickness maps overlaid on our own SHARAD interpretations in the MapView. We should be able to toggle between our analysis and the published results to see where they agree and where they differ.

**Yuri:** That's a great point, Elena. We should compare our results with the literature constantly. If our depth conversion doesn't match Petersen's or Bramson's, we need to know why. Is it our epsilon r assumption, or is there a local variation in the ice purity? Maybe the regolith on top is more dense in some areas.

**James:** I've been taking notes in the FieldNoteModal. I'll get these requirements into the next sprint. We can start with the dielectric calculator. It's a high priority for the science team.

**Anika:** I'll start looking at the traction models for ice rich regolith. We might need to update the SlopeAnalysis3DTab to account for different surface materials. I'll talk to the mechanical team about the wheel friction coefficients.

**Fatima:** And I'll refine the CNN classification for these ice associated minerals. We need to be able to distinguish between primary ice and the secondary minerals left behind. I'll use the Dundas sites as training data.

**Elena:** Excellent. We have a plan. Arcadia Planitia isn't just a landing site; it's a window into the Martian past. Let's make sure MarsLab is the best tool for looking through that window. We're pushing the boundaries of what's possible with orbital data.

**Yuri:** One last thing. When we present this to the board, let's make sure we emphasize the quantitative rigor. We aren't just looking at pretty pictures. We're measuring the permittivity of another world. We're using the physics of radar to map the hidden resources of Mars.

**James:** Spoken like a true geophysicist, Yuri. Meeting adjourned. I'll save the current state and share the session link with everyone. You can review the radargrams and the DTMs at your own pace.

**Marcus:** Don't forget to include the SHARAD track we analyzed. I want to double check those slopes one more time. I'll be looking at the 3D view to get a better sense of the terrain.

**Anika:** I'll send you the updated hazard map by tomorrow, Marcus. It will include the new ice proximity scores.

**Elena:** Great work, everyone. This is how we plan a mission. We're turning data into knowledge.