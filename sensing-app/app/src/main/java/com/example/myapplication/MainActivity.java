package com.example.myapplication;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Typeface;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.LocalBroadcastManager;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Gravity;
import android.view.Menu;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import java.util.List;

public class MainActivity extends AppCompatActivity implements ActivityCompat.OnRequestPermissionsResultCallback {

	private static final int PERMISSIONS_REQUEST_CODE_ACCESS_COARSE_LOCATION = 1;
	View mLayout;

	private final String TAG = "WifiScan";
	private WifiManager mWifiManager;
	private WifiData mWifiData;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		mLayout = findViewById(R.id.main_layout);
		mWifiData = new WifiData();
		mWifiManager = (WifiManager) this.getSystemService(Context.WIFI_SERVICE);

		// Register a listener for the 'Show Camera Preview' button.
		findViewById(R.id.button_scan_wifi).setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View view) {
				do_scan();
				setContentView(R.layout.activity_main);
				plotData();
			}
		});
	}

	private void getLocationPermission() {
		if (ActivityCompat.shouldShowRequestPermissionRationale(this,
				Manifest.permission.ACCESS_COARSE_LOCATION)) {

			Snackbar.make(mLayout, R.string.location_access_required,
					Snackbar.LENGTH_INDEFINITE).setAction(R.string.ok, new View.OnClickListener() {

				@Override
				public void onClick(View view) {
					// Request the permission
					ActivityCompat.requestPermissions(MainActivity.this,
							new String[]{Manifest.permission.ACCESS_COARSE_LOCATION},
							PERMISSIONS_REQUEST_CODE_ACCESS_COARSE_LOCATION);
				}
			}).show();
		} else {
			Snackbar.make(mLayout, R.string.location_unavailable, Snackbar.LENGTH_SHORT).show();
			// Request the permission. The result will be received in onRequestPermissionResult().
			ActivityCompat.requestPermissions(this,
					new String[]{Manifest.permission.ACCESS_COARSE_LOCATION}, PERMISSIONS_REQUEST_CODE_ACCESS_COARSE_LOCATION);
		}
	}

	private void scan() {
		List<ScanResult> mResults = mWifiManager.getScanResults();
		Log.d(TAG, "New scan result: (" + mResults.size() + ") networks found");
		mResults.toString();
		// store networks
		mWifiData.addNetworks(mResults);
		// send data to UI
		Intent intent = new Intent(Constants.INTENT_FILTER);
		intent.putExtra(Constants.WIFI_DATA, mWifiData);
		LocalBroadcastManager.getInstance(MainActivity.this).sendBroadcast(intent);
	}

	public void do_scan() {
		if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION)
				== PackageManager.PERMISSION_GRANTED) {
			// Permission is already available, start camera preview
			Snackbar.make(mLayout,
					R.string.location_permission_available,
					Snackbar.LENGTH_SHORT).show();
			scan();
		} else {
			// Permission is missing and must be requested.
			getLocationPermission();
		}

	}


	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		// Inflate the menu; this adds items to the action bar if it is present.
		getMenuInflater().inflate(R.menu.main, menu);
		return true;
	}

	@Override
	public void onPause() {
		super.onPause();
	}

	@Override
	public void onDestroy() {
		super.onDestroy();
	}


	public void plotData() {
		LinearLayout linearLayout = findViewById(R.id.scanningResultBlock);
		linearLayout.removeAllViews();

		if (mWifiData == null) {
			Log.d(TAG, "Plotting data: no networks");
			TextView noDataView = new TextView(this);
			noDataView.setText(getResources().getString(R.string.wifi_no_data));
			noDataView.setGravity(Gravity.CENTER_HORIZONTAL);
			noDataView.setPadding(0, 50, 0, 0);
			noDataView.setTextSize(18);
			linearLayout.addView(noDataView);
		} else {
			Log.d(TAG, "Plotting data");

			TableLayout.LayoutParams tableParams = new TableLayout.LayoutParams(TableLayout.LayoutParams.WRAP_CONTENT,
					TableLayout.LayoutParams.WRAP_CONTENT);
			TableRow.LayoutParams rowParams = new TableRow.LayoutParams(TableRow.LayoutParams.WRAP_CONTENT,
					TableRow.LayoutParams.WRAP_CONTENT);

			TableLayout tableLayout = new TableLayout(this);
			tableLayout.setLayoutParams(tableParams);
			tableLayout.setStretchAllColumns(true);

			// row header
			TableRow tableRowHeader = new TableRow(this);
			tableRowHeader.setLayoutParams(rowParams);

			TextView timeText = new TextView(this);
			timeText.setText(getResources().getString(R.string.time_text));
			timeText.setTypeface(null, Typeface.BOLD);

			TextView ssidText = new TextView(this);
			ssidText.setText(getResources().getString(R.string.ssid_text));
			ssidText.setTypeface(null, Typeface.BOLD);

			TextView chText = new TextView(this);
			chText.setText(getResources().getString(R.string.ch_text));
			chText.setTypeface(null, Typeface.BOLD);

			TextView rxText = new TextView(this);
			rxText.setText(getResources().getString(R.string.rx_text));
			rxText.setTypeface(null, Typeface.BOLD);

			TextView bssidText = new TextView(this);
			bssidText.setText(getResources().getString(R.string.bssid_text));
			bssidText.setTypeface(null, Typeface.BOLD);

			TextView freqText = new TextView(this);
			freqText.setText(getResources().getString(R.string.freq_text));
			freqText.setTypeface(null, Typeface.BOLD);

			tableRowHeader.addView(timeText);
			tableRowHeader.addView(ssidText);
			tableRowHeader.addView(bssidText);
			tableRowHeader.addView(chText);
			tableRowHeader.addView(rxText);
			tableRowHeader.addView(freqText);

			tableLayout.addView(tableRowHeader);

			// rows data
			for (WifiDataNetwork net : mWifiData.getNetworks()) {
				TextView time = new TextView(this);
				time.setText(String.valueOf(net.getTimestamp()));

				TextView ssidVal = new TextView(this);
				ssidVal.setText(net.getSsid());

				TextView chVal = new TextView(this);
				chVal.setText(String.valueOf(WifiDataNetwork.convertFrequencyToChannel(net.getFrequency())));

				TextView rxVal = new TextView(this);
				rxVal.setText(String.valueOf(net.getLevel()));

				TableRow tableRow = new TableRow(this);
				tableRow.setLayoutParams(rowParams);

				TextView bssidVal = new TextView(this);
				bssidVal.setText(net.getBssid());

				TextView freqVal = new TextView(this);
				freqVal.setText(String.valueOf(net.getFrequency()));
				//rxVal.setText(String.valueOf(net.getLevel()) + " dBm");

				tableRow.addView(time);
				tableRow.addView(ssidVal);
				tableRow.addView(bssidVal);
				tableRow.addView(chVal);
				tableRow.addView(rxVal);
				tableRow.addView(freqVal);

				tableLayout.addView(tableRow);
			}

			linearLayout.addView(tableLayout);
		}
	}

	public class MainActivityReceiver extends BroadcastReceiver {

		@Override
		public void onReceive(Context context, Intent intent) {
			WifiData data = intent.getParcelableExtra(Constants.WIFI_DATA);

			if (data != null) {
				mWifiData = data;
				plotData();
			}
		}

	}
}
