#pragma once

// Converts geodetic (WGS84 latitude/longitude/altitude) coordinates to a
// local East-North-Up (ENU) tangent-plane frame anchored at a fixed
// reference point, via the standard geodetic -> ECEF -> ENU pipeline (not a
// flat-Earth linear approximation -- at this project's track scale
// (hundreds of meters) the two agree closely, but ECEF is what a real GNSS
// receiver's own datasheet describes, so this is that, not a shortcut).
//
// Must be constructed with the SAME reference point as
// simulation/worlds/trackdrive.sdf's <spherical_coordinates> block, or the
// recovered ENU position won't correspond to Gazebo's own local world
// coordinates.
namespace fsd
{

struct EnuPosition
{
    double east;
    double north;
    double up;
};

class GeodeticConverter
{
public:
    GeodeticConverter(double _refLatDeg, double _refLonDeg, double _refAltM);

    EnuPosition ToEnu(double _latDeg, double _lonDeg, double _altM) const;

private:
    double m_refLatRad;
    double m_refLonRad;
    double m_refEcefX;
    double m_refEcefY;
    double m_refEcefZ;
};

} // namespace fsd
