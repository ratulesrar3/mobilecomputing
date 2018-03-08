package com.example.myapplication;

import java.util.List;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ListView;

public class MainActivity2  extends AppCompatActivity implements ActivityCompat.OnRequestPermissionsResultCallback {

	private WifiManager mainWifi;
	private WifiReceiver receiverWifi;
	private Button btnRefresh;
	ListAdapter adapter;
	ListView lvWifiDetails;
	List<ScanResult> wifiList;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		lvWifiDetails = (ListView) findViewById(R.id.lvWifiDetails);
		btnRefresh = (Button) findViewById(R.id.btnRefresh);
		mainWifi = (WifiManager) getSystemService(Context.WIFI_SERVICE);
		receiverWifi = new WifiReceiver();
		registerReceiver(receiverWifi, new IntentFilter(
				WifiManager.SCAN_RESULTS_AVAILABLE_ACTION));
		scanWifiList();

		btnRefresh.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {
				scanWifiList();

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

	private void scanWifiList() {
		mainWifi.startScan();
		wifiList = mainWifi.getScanResults();

		setAdapter();

	}

	class WifiReceiver extends BroadcastReceiver {
		public void onReceive(Context c, Intent intent) {
		}
	}
}
