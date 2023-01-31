const sampleRate = 16000;
// const domain = 'http://127.0.0.1:5000'
blobList = [];
filenameList = [];

jQuery(document).ready(function () {
	var $ = jQuery;
	var domain = $('#hostUrl').text();

	// $.ajax(domain + 'models', {
	// 	type: 'GET',
	// 	dataType: 'json',
	// })
	// .done((response) => {
	// 	if (response && response.data && response.data.length > 0) {
	// 		response.data.forEach(data => {
	// 			let optionObject = $(`<option value=${data.id}>${data.name}</option>`);
	// 			$('#model-selection').append(optionObject);
	// 		})
	// 	}
	// })
	// .fail((xhr, textStatus, errorThrown) => {
	// 	let errMsg = 'Failed retrieving models from backend. Error: ' + xhr.responseText;
	// 	console.log(errMsg);
	// 	alert(errMsg);
	// });

	let option1Object = $(`<option value=0>Best CNN Model</option>`);
	$('#model-selection').append(option1Object);
	let option2Object = $(`<option value=1>Final Model (CNN-LSTM)</option>`);
	$('#model-selection').append(option2Object);

	var myRecorder = {
		objects: {
			context: null,
			stream: null,
			recorder: null
		},
		init: function () {
			if (null === myRecorder.objects.context) {
				myRecorder.objects.context = new (
					window.AudioContext || window.webkitAudioContext
				);
			}
		},
		start: function () {
			var options = {audio: true, video: false};
			navigator.mediaDevices.getUserMedia(options).then(function (stream) {
				myRecorder.objects.stream = stream;
				myRecorder.objects.recorder = new Recorder(
					myRecorder.objects.context.createMediaStreamSource(stream),
					{numChannels: 1}
				);
				myRecorder.objects.recorder.record();
			}).catch(function (err) {});
		},
		stop: function (listObject) {
			if (null !== myRecorder.objects.stream) {
				myRecorder.objects.stream.getAudioTracks()[0].stop();
			}
			if (null !== myRecorder.objects.recorder) {
				myRecorder.objects.recorder.stop();

				// Validate object
				if (null !== listObject
						&& 'object' === typeof listObject
						&& listObject.length > 0) {
					// Export the WAV file
					myRecorder.objects.recorder.exportWAV(function (blob) {
						var filename = new Date().toLocaleString('en-US', {
																	timeZone: 'Hongkong'
																})
																.replaceAll(',', '')
																.replaceAll('/', '-')
																.replace(':', 'h')
																.replace(':', 'm');
						
						addAudioRow(filename + '.wav', blob);
					});
				}
			}
		}
	};

	// Prepare the recordings list
	var listObject = $('[data-role="recordings"]');

	// Prepare the record button
	$('[data-role="controls"] > button').click(function (e) {
		e.preventDefault();
		e.stopPropagation();

		// Initialize the recorder
		myRecorder.init();

		// Get the button state 
		var buttonState = !!$(this).attr('data-recording');

		// Toggle
		if (!buttonState) {
			$(this).attr('data-recording', 'true');
			myRecorder.start();
		} else {
			$(this).attr('data-recording', '');
			myRecorder.stop(listObject);
		}
	});

	$('#close-mel-spectrogram').click((e) => {
		e.preventDefault();
		e.stopPropagation();

		$('#mel-spectrogram-section').attr('show', 'False');
		$('#close-mel-spectrogram').nextAll().remove();
	})


	// Console Log Audio Data List
	$('[data-role="predict-emotion-button"]').click((e) => {
		e.preventDefault();
		e.stopPropagation();

		loadScreen(true);

		const url = domain + 'predict';

		const modelChoice = $('#model-selection').val();
		
		var formData = new FormData();
		formData.append('modelChoice', modelChoice);
		for (let i = 0; i < blobList.length; i++) {
			const blob = blobList[i];
			const filename = filenameList[i];
			const filenameNoExtension = filename.substring(0, filename.indexOf('.'));

			formData.append(filenameNoExtension, blob, filename);
		}
		

		$.ajax(url, {
			type: 'POST',
			dataType: 'json',
			data: formData,
			cache: false,
			contentType: false,
			processData: false,
		})
		.done((response) => {
			// Clear Previous Result
			$('ul.emotion-result').empty();

			loadScreen(false);

			// Show Predicted Result
			if (response && response.status) {
				if (response.status == 'ok') {
					if (response.data.length > 0) {
						response.data.forEach(data => {
							let emotion = data.emotion;
							let name = data.name.replaceAll('.wav', '').replaceAll(' ', '-');
							let section = data.section;
							let sectionClass = section.replaceAll(':', '').replaceAll(' ', '');
							let percentage = data.percentage;
							let anger_percentage = percentage.Anger ? parseFloat(percentage.Anger) * 100 : 0.0;
							let excitement_percentage = percentage.Excitement ? parseFloat(percentage.Excitement) * 100 : 0.0;
							let frustration_percentage = percentage.Frustration ? parseFloat(percentage.Frustration) * 100 : 0.0;
							let happiness_percentage = percentage.Happiness ? parseFloat(percentage.Happiness) * 100 : 0.0;
							let sadness_percentage = percentage.Sadness ? parseFloat(percentage.Sadness) * 100 : 0.0;
							let neutral_percentage = percentage.Neutral ? parseFloat(percentage.Neutral) * 100 : 0.0;

							let colorA = '#8B8484'; // Darker
							let colorB = '#B8B8B8'; // Lighter

							if (emotion === 'Anger') {
								// Red
								colorA = '#E61E1E';
								colorB = '#E18F89';
							}
							else if (emotion == 'Excitement') {
								// Yellow
								colorA = '#EBE300';
								colorB = '#CBD080';
							}
							else if (emotion == 'Frustration') {
								// Purple
								colorA = '#AD01A7';
								colorB = '#D67AC5';
							}
							else if (emotion == 'Happiness') {
								// Green
								colorA = '#00EB46';
								colorB = '#76D0A2';
							}
							else if (emotion == 'Neutral') {
								// Grey
								colorA = '#AFBBB5';
								colorB = '#DAE1DE';
							}
							else if (emotion == 'Sadness') {
								// Blue
								colorA = '#0093E9';
								colorB = '#80D0C7';
							}
							let style = `background-color: ${colorA};background-image: linear-gradient(60deg, ${colorA} 0%, ${colorB} 100%);`
							let resultObject = $(`
								<li class="emotion-container">
									<div class="emotion-section-result ${sectionClass}" style="${style}">
										<span class="time">${section}</span>
										<span class="emotion">${emotion}</span>
									</div>
									<div class="emotion-percentages">
										<div class="specific-emotion-percentage anger-emotion" style="width: ${anger_percentage}%;">${anger_percentage >= 10 ? anger_percentage.toFixed(1).toString() + "%" : ""}</div>
										<div class="specific-emotion-percentage excitement-emotion" style="width: ${excitement_percentage}%;">${excitement_percentage >= 10 ? excitement_percentage.toFixed(1).toString() + "%" : ""}</div>
										<div class="specific-emotion-percentage frustration-emotion" style="width: ${frustration_percentage}%;">${frustration_percentage >= 10 ? frustration_percentage.toFixed(1).toString() + "%" : ""}</div>
										<div class="specific-emotion-percentage happiness-emotion" style="width: ${happiness_percentage}%;">${happiness_percentage >= 10 ? happiness_percentage.toFixed(1).toString() + "%" : ""}</div>
										<div class="specific-emotion-percentage sadness-emotion" style="width: ${sadness_percentage}%;">${sadness_percentage >= 10 ? sadness_percentage.toFixed(1).toString() + "%" : ""}</div>											
										<div class="specific-emotion-percentage neutral-emotion" style="width: ${neutral_percentage}%;">${neutral_percentage >= 10 ? neutral_percentage.toFixed(1).toString() + "%" : ""}</div>
									</div>
								</li>
							`);
							$(`ul#${name}`).append(resultObject);

						});
					}
				}
				else if (response.status == 'failed') {
					console.log(response.errMsg);
					alert(response.errMsg);
				}
				else if (response.status == 'warning') {
					console.log(response.errMsg);
				}
			}
			
		})
		.fail((xhr, textStatus, errorThrown) => {
			loadScreen(false);

			let errMsg = 'Request failed with error: ' + xhr.responseText;
			console.log(errMsg);
			alert(errMsg);
		});
	})

	// File Upload Section
	$('#upload-form').click((e) => {
		e.preventDefault();
		e.stopPropagation();

		$('#file-input').trigger('click');
	})

	$('#file-input').change((e) => {
		e.preventDefault();
		e.stopPropagation();

		if (e.target.files && e.target.files.length > 0) {
			const files = e.target.files;
			storeFiles(files);
		}
	})

	 // preventing page from redirecting
	 $('#upload-form').on('dragover', e => {
		e.preventDefault();
		e.stopPropagation();
		
		$('#upload-form').attr('drop-active', 'True');
	 });

	$('#upload-form').on('dragleave', e => {
		e.preventDefault();
		e.stopPropagation();
		console.log('leave');

		$('#upload-form').attr('drop-active', 'False');
	})
	
	 $('#upload-form').on('drop', e => {
		e.preventDefault();
		e.stopPropagation();

		$('#upload-form').attr('drop-active', 'False');

		const files = e.originalEvent.dataTransfer.files;
		storeFiles(files);
	});

	var storeFiles = (files) => {
		for (let i = 0; i < files.length; i++) {
			const file = files[i];
			// if (file.type != 'audio/wav' || file.type != 'audio/x-m4a' || file.type != 'audio/mpeg'  || file.type != 'audio/ogg')
			if (file.type !== 'audio/wav' && file.type !== 'audio/x-m4a'
				&& file.type !== 'audio/mpeg' && file.type !== 'audio/ogg'
				&& file.type !== 'audio/basic') {
				
				let errMsg = "Please only upload .wav, .m4a, .mp3, .ogg, .opus, or .au file type!";
				console.log(errMsg);
				alert(errMsg);
			}
			else {
				// let dateName = new Date().toLocaleString('en-US', {
				// 												timeZone: 'Hongkong'
				// 											})
				// 											.replaceAll(',', '')
				// 											.replaceAll('/', '-')
				// 											.replace(':', 'h')
				// 											.replace(':', 'm');
				
				// let filename = file.name.substring(file.name.indexOf('.'));
				// filename = dateName + filename;
				addAudioRow(file.name, file);
			}
		}
	}

	var addAudioRow = (filename, blob) => {
		var url = (window.URL || window.webkitURL)
				.createObjectURL(blob);

		// Prepare the playback
		var audioObject = $('<audio controls></audio>')
				.attr('src', url);

		// Prepare the download link
		var downloadObject = $('<a class="download">&#9660;</a>')
				.attr('href', url)
				.attr('download', filename);
		
		var expandObject = $('<a class="expand">&#x2B;</a>')
		expandObject.click(() => {

			loadScreen(true);

			const modelChoice = $('#model-selection').val();

			var formData = new FormData();
			formData.append('modelChoice', modelChoice);
			formData.append('dataFileName', filename);

			$.ajax(domain + 'mel-spectrogram', {
				type: 'POST',
				dataType: 'json',
				data: formData,
				cache: false,
				contentType: false,
				processData: false,
			})
			.done((response) => {
				loadScreen(false);

				if (response && response.data) {
					if (response.status == 'ok') {
						if (response.data.length > 0) {
							console.log(response.data);
							for (let i = 0; i < response.data.length; i++) {
								let data_object = response.data[i];
								console.log(data_object.mel_spectrogram);
								// let decoded = btoa(String.fromCharCode.apply(null, data_object.replace(/\r|\n/g, "").replace(/([\da-fA-F]{2}) ?/g, "0x$1 ").replace(/ +$/, "").split(" ")))
								let imageSrc = 'data:image/png;base64,' + data_object.mel_spectrogram;

								var predictedContainerObject = $('<div class="predicted-container"></div>');
								var predictedEmotionObject = $(`<h2 class="predicted-emotion">${data_object.emotion}</h2>`)
								var audioFileNameObject = $(`<h3 class="audio-filename">${data_object.name} ${data_object.section}</h3>`);
								var melSpectrogramImageContainerObject = $(`<div class="mel-spectrogram-image-container"></div>`);
								var imageObject = $('<img alt="mel spectrogram image" class="mel-spectrogram-image" />')
											.attr('src', imageSrc);
								
								melSpectrogramImageContainerObject.append(imageObject);

								predictedContainerObject.append(predictedEmotionObject)
																				.append(audioFileNameObject)
																				.append(melSpectrogramImageContainerObject);
								
								$('#mel-spectrogram-section').append(predictedContainerObject);
								
								$('#mel-spectrogram-section').attr('show', 'True');
								
								
							}
							
						}
						else {
							errMsg = 'No mel-spectrogram image in response data';
							console.log(errMsg);
							alert(errMsg);
						}
					}
					else if (response.status == 'failed') {
						console.log(response.errMsg);
						alert(response.errMsg);
					}
					else if (response.status == 'warning') {
						console.log(response.errMsg);
					}
				}
			})
			.fail((xhr, textStatus, errorThrown) => {
				loadScreen(false);

				let errMsg = 'Failed retrieving mel spectrogram images from backend. Error: ' + xhr.responseText;
				console.log(errMsg);
				alert(errMsg);
			});
		})
		
		let classFileName = filename.replaceAll(' ', '-');
		classFileName = classFileName.substring(0, classFileName.indexOf('.'));
		var emotionObject = $(`<ul id="${classFileName}" class="emotion-result"></ul>`);
	
		var audioName = $(`<div class="audio-name">${filename}</div>`)

		// Wrap everything in a row
		var holderObject = $('<div class="audio-holder"></div>')
						.append(audioObject);

		var audioRowObject = $('<div class="audio-row"></div>')
						.append(audioName)
						.append(holderObject)
						.append(downloadObject)
						.append(expandObject)
						.append(emotionObject);
		


		// Append to the list
		listObject.append(audioRowObject);

		blobList.push(blob);
		filenameList.push(filename);
	}

	var loadScreen = (isLoading) => {
		if (isLoading) {
			$('#loading').attr('loading-active', 'True')
		}
		else {
			$('#loading').attr('loading-active', 'False')
		}
	}
});
