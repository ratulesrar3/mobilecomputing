package com.example.myapplication;

import java.util.List;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.LocalBroadcastManager;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ListView;

import java.util.List;

public class MainActivity extends AppCompatActivity implements ActivityCompat.OnRequestPermissionsResultCallback {

	private static final int PERMISSIONS_REQUEST_CODE_ACCESS_COARSE_LOCATION = 1;
	View mLayout;

	private final String TAG = "WifiScan";
	private WifiManager mWifiManager;
	private WifiData mWifiData;

	private WifiManager mainWifi;
	private MainActivity2.WifiReceiver receiverWifi;
	private Button btnRefresh;
	ListAdapter adapter;
	ListView lvWifiDetails;
	List<ScanResult> wifiList;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		lvWifiDetails = (ListView) findViewById(R.id.lvWifiDetails);

		mainWifi = (WifiManager) getSystemService(Context.WIFI_SERVICE);
		receiverWifi = new MainActivity2.WifiReceiver();
		registerReceiver(receiverWifi, new IntentFilter(
				WifiManager.SCAN_RESULTS_AVAILABLE_ACTION));
		doScan();

		findViewById(R.id.button_scan_wifi).setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {
				doScan();

			}
		});
	}

	private void setAdapter() {
		adapter = new ListAdapter(getApplicationContext(), wifiList);
		lvWifiDetails.setAdapter(adapter);
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
		mainWifi.startScan();
		wifiList = mainWifi.getScanResults();

		setAdapter();

		/*List<ScanResult> mResults = mWifiManager.getScanResults();
		Log.d(TAG, "New scan result: (" + mResults.size() + ") networks found");
		mResults.toString();
		// store networks
		mWifiData.addNetworks(mResults);
		// send data to UI
		Intent intent = new Intent(Constants.INTENT_FILTER);
		intent.putExtra(Constants.WIFI_DATA, mWifiData);
		LocalBroadcastManager.getInstance(MainActivity.this).sendBroadcast(intent);*/
	}

	public void doScan() {
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


	class WifiReceiver extends BroadcastReceiver {
		public void onReceive(Context c, Intent intent) {
		}




}
