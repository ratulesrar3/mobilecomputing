package com.example.myapplication;

import java.util.List;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.view.View;

import android.net.wifi.ScanResult;
import android.net.wifi.WifiManager;
import android.support.v4.content.LocalBroadcastManager;
import android.util.Log;

public class WifiScan extends AppCompatActivity implements ActivityCompat.OnRequestPermissionsResultCallback {

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
            }
        });
    }

    /**
     * Performs a periodical read of the WiFi scan result, then it creates a new
     * {@link WifiData()} object containing the list of networks and finally it
     * sends it to the main activity for being displayed.
     */
    private void getLocationPermission() {
        if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                Manifest.permission.ACCESS_COARSE_LOCATION)) {

            Snackbar.make(mLayout, R.string.location_access_required,
                    Snackbar.LENGTH_INDEFINITE).setAction(R.string.ok, new View.OnClickListener() {

                @Override
                public void onClick(View view) {
                    // Request the permission
                    ActivityCompat.requestPermissions(WifiScan.this,
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
        // store networks
        mWifiData.addNetworks(mResults);
        // send data to UI
        Intent intent = new Intent(Constants.INTENT_FILTER);
        intent.putExtra(Constants.WIFI_DATA, mWifiData);
        LocalBroadcastManager.getInstance(WifiScan.this).sendBroadcast(intent);
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
}

